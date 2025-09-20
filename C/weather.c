// weather.c — volumetric weather with slice-based rendering over terrain
// Build (macOS/Homebrew):
// clang -std=c11 -O2 weather.c -o weather \
//   -I/opt/homebrew/include -L/opt/homebrew/lib -lraylib \
//   -framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL

#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// --- small helpers ---
static inline float clampf(float v, float lo, float hi) { return (v < lo) ? lo : (v > hi ? hi : v); }
static inline int   clampi(int   v, int   lo, int   hi) { return (v < lo) ? lo : (v > hi ? hi : v); }

// ---------- Domain & grid ----------
#define TERRAIN_SIZE_X 256
#define TERRAIN_SIZE_Z 256

#define NX 48
#define NY 32
#define NZ 48

#define WORLD_TOP_Y 160.0f

static const float DX = (float)TERRAIN_SIZE_X / (float)NX;
static const float DZ = (float)TERRAIN_SIZE_Z / (float)NZ;
static const float DY = WORLD_TOP_Y / (float)NY;

// ---------- Simulation tuning ----------
static float g_dt = 0.08f;
static int   g_substeps = 1;
static float g_visc = 0.0008f;
static float g_diff = 0.0005f;
static float g_buoyancyT = 0.80f;
static float g_buoyancyH = 0.10f;
static float g_windInflow = 0.8f;
static float g_inflowHumidity = 0.60f;   // moist air at -X inflow
static float g_groundEvap = 0.020f;      // near-ground evaporation per second
static float g_evapHeight = 2.0f;        // in world units
static int   g_jacobiIters = 18;

// ---------- Visualization ----------
static int   g_mode = 1; // 1=temp, 2=humidity, 3=clouds
static int   g_drawStep = 2; // cube debug step
static bool  g_wireframeTerrain = false;
static bool  g_debugCubes = false; // toggle cubes

// ---------- Terrain heightmap ----------
static float terrainH[TERRAIN_SIZE_X][TERRAIN_SIZE_Z];
static float terrainMinY = 0.0f, terrainMaxY = 0.0f;

static bool LoadTerrainHeightFromOBJ(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return false;
    for (int z=0; z<TERRAIN_SIZE_Z; ++z)
        for (int x=0; x<TERRAIN_SIZE_X; ++x)
            terrainH[x][z] = 0.0f;
    terrainMinY = 1e9f; terrainMaxY = -1e9f;

    char line[512];
    int cnt = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0]=='v' && line[1]==' ') {
            float vx, vy, vz;
            if (sscanf(line, "v %f %f %f", &vx, &vy, &vz) == 3) {
                int ix = (int)floorf(vx + 0.5f);
                int iz = (int)floorf(vz + 0.5f);
                if (ix>=0 && ix<TERRAIN_SIZE_X && iz>=0 && iz<TERRAIN_SIZE_Z) {
                    if (vy > terrainH[ix][iz]) terrainH[ix][iz] = vy;
                    if (vy < terrainMinY) terrainMinY = vy;
                    if (vy > terrainMaxY) terrainMaxY = vy;
                    cnt++;
                }
            }
        }
    }
    fclose(f);
    return cnt > 0;
}

// Bilinear height sample in world coords
static inline float SampleTerrainHeightBilinear(float wx, float wz) {
    float x = clampf(wx, 0.0f, (float)TERRAIN_SIZE_X - 1.0f);
    float z = clampf(wz, 0.0f, (float)TERRAIN_SIZE_Z - 1.0f);
    int x0 = (int)floorf(x), z0 = (int)floorf(z);
    int x1 = (x0+1 < TERRAIN_SIZE_X) ? x0+1 : x0;
    int z1 = (z0+1 < TERRAIN_SIZE_Z) ? z0+1 : z0;
    float fx = x - x0, fz = z - z0;
    float h00 = terrainH[x0][z0], h10 = terrainH[x1][z0];
    float h01 = terrainH[x0][z1], h11 = terrainH[x1][z1];
    float hx0 = h00*(1.0f-fx) + h10*fx;
    float hx1 = h01*(1.0f-fx) + h11*fx;
    return hx0*(1.0f-fz) + hx1*fz;
}

// Build a shaded terrain model from heightmap (robust vs OBJ loader)
static Model BuildTerrainModelFromHeight(void) {
    const int W=TERRAIN_SIZE_X, H=TERRAIN_SIZE_Z;
    const int vcount=W*H;
    const int icount=(W-1)*(H-1)*6;

    Mesh mesh = (Mesh){0};
    mesh.vertexCount = vcount;
    mesh.triangleCount = icount/3;
    mesh.vertices = (float*)MemAlloc(vcount*3*sizeof(float));
    mesh.normals  = (float*)MemAlloc(vcount*3*sizeof(float));
    mesh.colors   = (unsigned char*)MemAlloc(vcount*4*sizeof(unsigned char));
    mesh.indices  = (unsigned short*)MemAlloc(icount*sizeof(unsigned short));

    for (int z=0; z<H; ++z) for (int x=0; x<W; ++x) {
        int i = z*W + x;
        float y = terrainH[x][z];
        mesh.vertices[i*3+0] = (float)x;
        mesh.vertices[i*3+1] = y;
        mesh.vertices[i*3+2] = (float)z;
        mesh.normals[i*3+0] = mesh.normals[i*3+1] = mesh.normals[i*3+2] = 0.0f;
    }
    int idx=0;
    for (int z=0; z<H-1; ++z) for (int x=0; x<W-1; ++x) {
        int i0=z*W+x, i1=i0+1, i2=i0+W, i3=i2+1;
        mesh.indices[idx++]=i0; mesh.indices[idx++]=i2; mesh.indices[idx++]=i1;
        mesh.indices[idx++]=i1; mesh.indices[idx++]=i2; mesh.indices[idx++]=i3;
    }
    for (int t=0; t<icount; t+=3) {
        int i0=mesh.indices[t], i1=mesh.indices[t+1], i2=mesh.indices[t+2];
        Vector3 v0={mesh.vertices[i0*3+0],mesh.vertices[i0*3+1],mesh.vertices[i0*3+2]};
        Vector3 v1={mesh.vertices[i1*3+0],mesh.vertices[i1*3+1],mesh.vertices[i1*3+2]};
        Vector3 v2={mesh.vertices[i2*3+0],mesh.vertices[i2*3+1],mesh.vertices[i2*3+2]};
        Vector3 n = Vector3Normalize(Vector3CrossProduct(Vector3Subtract(v1,v0), Vector3Subtract(v2,v0)));
        mesh.normals[i0*3+0]+=n.x; mesh.normals[i0*3+1]+=n.y; mesh.normals[i0*3+2]+=n.z;
        mesh.normals[i1*3+0]+=n.x; mesh.normals[i1*3+1]+=n.y; mesh.normals[i1*3+2]+=n.z;
        mesh.normals[i2*3+0]+=n.x; mesh.normals[i2*3+1]+=n.y; mesh.normals[i2*3+2]+=n.z;
    }
    for (int i=0; i<vcount; ++i) {
        Vector3 n = Vector3Normalize((Vector3){mesh.normals[i*3+0],mesh.normals[i*3+1],mesh.normals[i*3+2]});
        mesh.normals[i*3+0]=n.x; mesh.normals[i*3+1]=n.y; mesh.normals[i*3+2]=n.z;
        float shade = Clamp(Vector3DotProduct(n, Vector3Normalize((Vector3){0.35f,1.0f,0.45f})), 0.0f, 1.0f);
        float slope = 1.0f - Vector3DotProduct(n,(Vector3){0,1,0}); slope = clampf(slope,0.0f,1.0f);
        Color base = (slope > 0.35f)?(Color){110,95,85,255}:(Color){55,120,50,255};
        mesh.colors[i*4+0]=(unsigned char)(base.r*(0.3f+0.7f*shade));
        mesh.colors[i*4+1]=(unsigned char)(base.g*(0.3f+0.7f*shade));
        mesh.colors[i*4+2]=(unsigned char)(base.b*(0.3f+0.7f*shade));
        mesh.colors[i*4+3]=255;
    }
    UploadMesh(&mesh, true);
    return LoadModelFromMesh(mesh);
}

// ---------- Fields ----------
static float *U,*V,*W,*U0,*V0,*W0,*T,*T0,*H,*H0,*C,*C0,*P,*Div;
static unsigned char *Solid;

static inline int idx3(int i,int j,int k){ return i + NX*(k + NZ*j); }

static void AllocateFields(void){
    size_t N=(size_t)NX*NY*NZ;
    U=MemAlloc(N*sizeof(float)); V=MemAlloc(N*sizeof(float)); W=MemAlloc(N*sizeof(float));
    U0=MemAlloc(N*sizeof(float)); V0=MemAlloc(N*sizeof(float)); W0=MemAlloc(N*sizeof(float));
    T=MemAlloc(N*sizeof(float)); T0=MemAlloc(N*sizeof(float));
    H=MemAlloc(N*sizeof(float)); H0=MemAlloc(N*sizeof(float));
    C=MemAlloc(N*sizeof(float)); C0=MemAlloc(N*sizeof(float));
    P=MemAlloc(N*sizeof(float)); Div=MemAlloc(N*sizeof(float));
    Solid=MemAlloc(N*sizeof(unsigned char));
}
static void FreeFields(void){
    MemFree(U); MemFree(V); MemFree(W);
    MemFree(U0); MemFree(V0); MemFree(W0);
    MemFree(T); MemFree(T0);
    MemFree(H); MemFree(H0);
    MemFree(C); MemFree(C0);
    MemFree(P); MemFree(Div);
    MemFree(Solid);
}

static void ResetFields(void){
    for(int j=0;j<NY;++j) for(int k=0;k<NZ;++k) for(int i=0;i<NX;++i){
        int id=idx3(i,j,k);
        float x=(i+0.5f)*DX, y=(j+0.5f)*DY, z=(k+0.5f)*DZ;
        float terr = SampleTerrainHeightBilinear(x,z);
        Solid[id] = (y <= terr) ? 1 : 0;

        U[id]=V[id]=W[id]=0.0f; P[id]=0.0f;

        float tBase = 0.85f - 0.65f*(y/WORLD_TOP_Y);
        float valleyBoost = (terr < (terrainMinY + 0.25f*(terrainMaxY-terrainMinY)))? 0.04f : 0.0f;
        T[id] = clampf(tBase + valleyBoost, 0.0f, 1.0f);

        float hBase = 0.35f - 0.25f*(y/WORLD_TOP_Y);
        float valleyHum = (terr < (terrainMinY + 0.35f*(terrainMaxY-terrainMinY)))? 0.25f : 0.0f;
        H[id] = clampf(hBase + valleyHum, 0.0f, 1.0f);
        C[id] = 0.0f;
    }
}

// ---------- Boundary/obstacles ----------
static void EnforceSolids(float *u,float *v,float *w){
    for(int j=0;j<NY;++j) for(int k=0;k<NZ;++k) for(int i=0;i<NX;++i){
        int id=idx3(i,j,k);
        if(Solid[id]) { u[id]=v[id]=w[id]=0.0f; continue; }
        if(i>0    && Solid[idx3(i-1,j,k)]) u[id]=fminf(u[id],0.0f);
        if(i<NX-1 && Solid[idx3(i+1,j,k)]) u[id]=fmaxf(u[id],0.0f);
        if(j>0    && Solid[idx3(i,j-1,k)]) v[id]=fminf(v[id],0.0f);
        if(j<NY-1 && Solid[idx3(i,j+1,k)]) v[id]=fmaxf(v[id],0.0f);
        if(k>0    && Solid[idx3(i,j,k-1)]) w[id]=fminf(w[id],0.0f);
        if(k<NZ-1 && Solid[idx3(i,j,k+1)]) w[id]=fmaxf(w[id],0.0f);
    }
    for(int j=0;j<NY;++j) for(int k=0;k<NZ;++k) {
        U[idx3(0,j,k)]=fmaxf(U[idx3(0,j,k)],0.0f);
        U[idx3(NX-1,j,k)]=fminf(U[idx3(NX-1,j,k)],0.0f);
    }
    for(int i=0;i<NX;++i) for(int k=0;k<NZ;++k) {
        V[idx3(i,0,k)]=fmaxf(V[idx3(i,0,k)],0.0f);
        V[idx3(i,NY-1,k)]=fminf(V[idx3(i,NY-1,k)],0.0f);
    }
    for(int i=0;i<NX;++i) for(int j=0;j<NY;++j) {
        W[idx3(i,j,0)]=fmaxf(W[idx3(i,j,0)],0.0f);
        W[idx3(i,j,NZ-1)]=fminf(W[idx3(i,j,NZ-1)],0.0f);
    }
}

// ---------- Operators ----------
static void AddBuoyancy(float *v,float *temp,float *hum,float dt){
    for(int id=0; id<NX*NY*NZ; ++id){
        if(Solid[id]) continue;
        float force = g_buoyancyT*(temp[id]-0.4f) - g_buoyancyH*hum[id];
        v[id] += force * dt;
    }
}

static void AddInflow(float *u, float *hum){
    // velocity inflow
    for (int j=0;j<NY;++j) for (int k=0;k<NZ;++k) for (int i=0;i<2;++i) {
        int id=idx3(i,j,k);
        if(!Solid[id]) u[id]=fmaxf(u[id], g_windInflow);
    }
    // humidity inflow (keep moist at boundary)
    for (int j=0;j<NY;++j) for (int k=0;k<NZ;++k) {
        int id=idx3(0,j,k);
        if(!Solid[id]) hum[id] = fmaxf(hum[id], g_inflowHumidity);
    }
}

static void AddGroundEvaporation(float *hum, float dt){
    for(int j=0;j<NY;++j) for(int k=0;k<NZ;++k) for(int i=0;i<NX;++i){
        int id=idx3(i,j,k);
        if(Solid[id]) continue;
        float x=(i+0.5f)*DX, y=(j+0.5f)*DY, z=(k+0.5f)*DZ;
        float terr = SampleTerrainHeightBilinear(x,z);
        float d = y - terr;
        if (d>0.0f && d < g_evapHeight) {
            float w = 1.0f - (d / g_evapHeight); // stronger at ground
            hum[id] = clampf(hum[id] + g_groundEvap * w * dt, 0.0f, 1.0f);
        }
    }
}

static void Diffuse(float *dst,const float *src,float diff,float dt,int iters,int isVel){
    float a = dt*diff/(DX*DX);
    for(int id=0; id<NX*NY*NZ; ++id) dst[id]=src[id];
    for(int it=0; it<iters; ++it){
        for(int j=1;j<NY-1;++j) for(int k=1;k<NZ-1;++k) for(int i=1;i<NX-1;++i){
            int id=idx3(i,j,k);
            if(Solid[id]){ dst[id]=0.0f; continue; }
            float sum = dst[idx3(i-1,j,k)]+dst[idx3(i+1,j,k)]
                      + dst[idx3(i,j-1,k)]+dst[idx3(i,j+1,k)]
                      + dst[idx3(i,j,k-1)]+dst[idx3(i,j,k+1)];
            dst[id]=(src[id]+a*sum)/(1.0f+6.0f*a);
        }
        if(isVel) EnforceSolids(dst,dst,dst);
    }
}

static inline float sampleScalar(const float* s, float x,float y,float z){
    int i0=clampi((int)floorf(x),0,NX-1), j0=clampi((int)floorf(y),0,NY-1), k0=clampi((int)floorf(z),0,NZ-1);
    int i1=clampi(i0+1,0,NX-1), j1=clampi(j0+1,0,NY-1), k1=clampi(k0+1,0,NZ-1);
    float fx=clampf(x-i0,0,1), fy=clampf(y-j0,0,1), fz=clampf(z-k0,0,1);
    float c000=s[idx3(i0,j0,k0)], c100=s[idx3(i1,j0,k0)];
    float c010=s[idx3(i0,j1,k0)], c110=s[idx3(i1,j1,k0)];
    float c001=s[idx3(i0,j0,k1)], c101=s[idx3(i1,j0,k1)];
    float c011=s[idx3(i0,j1,k1)], c111=s[idx3(i1,j1,k1)];
    float c00=c000*(1-fx)+c100*fx, c10=c010*(1-fx)+c110*fx;
    float c01=c001*(1-fx)+c101*fx, c11=c011*(1-fx)+c111*fx;
    float c0=c00*(1-fy)+c10*fy, c1=c01*(1-fy)+c11*fy;
    return c0*(1-fz)+c1*fz;
}
static inline Vector3 sampleVelocity(const float* u,const float* v,const float* w,float x,float y,float z){
    return (Vector3){ sampleScalar(u,x,y,z), sampleScalar(v,x,y,z), sampleScalar(w,x,y,z) };
}

static void AdvectScalarField(float *dst,const float *src,const float *u,const float *v,const float *w,float dt){
    for(int j=0;j<NY;++j) for(int k=0;k<NZ;++k) for(int i=0;i<NX;++i){
        int id=idx3(i,j,k);
        if(Solid[id]){ dst[id]=0.0f; continue; }
        float vx=u[id]/DX, vy=v[id]/DY, vz=w[id]/DZ;
        float x=i - dt*vx, y=j - dt*vy, z=k - dt*vz;
        dst[id]=sampleScalar(src, clampf(x,0,NX-1), clampf(y,0,NY-1), clampf(z,0,NZ-1));
    }
}
static void AdvectVelocityField(float *un,float *vn,float *wn,const float *u,const float *v,const float *w,float dt){
    for(int j=0;j<NY;++j) for(int k=0;k<NZ;++k) for(int i=0;i<NX;++i){
        int id=idx3(i,j,k);
        if(Solid[id]){ un[id]=vn[id]=wn[id]=0.0f; continue; }
        float vx=u[id]/DX, vy=v[id]/DY, vz=w[id]/DZ;
        float x=i - dt*vx, y=j - dt*vy, z=k - dt*vz;
        Vector3 vel=sampleVelocity(u,v,w, clampf(x,0,NX-1), clampf(y,0,NY-1), clampf(z,0,NZ-1));
        un[id]=vel.x; vn[id]=vel.y; wn[id]=vel.z;
    }
    EnforceSolids(un,vn,wn);
}

static void ComputeDivergence(const float *u,const float *v,const float *w){
    const float inv2dx=0.5f/DX, inv2dy=0.5f/DY, inv2dz=0.5f/DZ;
    for(int j=1;j<NY-1;++j) for(int k=1;k<NZ-1;++k) for(int i=1;i<NX-1;++i){
        int id=idx3(i,j,k);
        if(Solid[id]){ Div[id]=0.0f; continue; }
        float du=u[idx3(i+1,j,k)]-u[idx3(i-1,j,k)];
        float dv=v[idx3(i,j+1,k)]-v[idx3(i,j-1,k)];
        float dw=w[idx3(i,j,k+1)]-w[idx3(i,j,k-1)];
        Div[id]=inv2dx*du + inv2dy*dv + inv2dz*dw;
    }
}
static void SolvePressure(void){
    for(int id=0; id<NX*NY*NZ; ++id) if(Solid[id]) P[id]=0.0f;
    for(int it=0; it<g_jacobiIters; ++it){
        for(int j=1;j<NY-1;++j) for(int k=1;k<NZ-1;++k) for(int i=1;i<NX-1;++i){
            int id=idx3(i,j,k);
            if(Solid[id]){ P[id]=0.0f; continue; }
            float sum=P[idx3(i-1,j,k)]+P[idx3(i+1,j,k)]
                     +P[idx3(i,j-1,k)]+P[idx3(i,j+1,k)]
                     +P[idx3(i,j,k-1)]+P[idx3(i,j,k+1)];
            P[id]=(sum - Div[id])/6.0f;
        }
    }
}
static void SubtractPressureGradient(float *u,float *v,float *w){
    const float inv2dx=0.5f/DX, inv2dy=0.5f/DY, inv2dz=0.5f/DZ;
    for(int j=1;j<NY-1;++j) for(int k=1;k<NZ-1;++k) for(int i=1;i<NX-1;++i){
        int id=idx3(i,j,k);
        if(Solid[id]) continue;
        float dpdx=(P[idx3(i+1,j,k)]-P[idx3(i-1,j,k)])*inv2dx;
        float dpdy=(P[idx3(i,j+1,k)]-P[idx3(i,j-1,k)])*inv2dy;
        float dpdz=(P[idx3(i,j,k+1)]-P[idx3(i,j,k-1)])*inv2dz;
        u[id]-=dpdx; v[id]-=dpdy; w[id]-=dpdz;
    }
    EnforceSolids(u,v,w);
}

// Tweaked condensation (more visible clouds)
static void Condense(float *temp,float *hum,float *cloud,float dt){
    for(int id=0; id<NX*NY*NZ; ++id){
        if(Solid[id]){ cloud[id]=0.0f; continue; }
        float Tn = temp[id];
        float sat = clampf(0.45f*Tn + 0.20f, 0.12f, 0.70f);  // lower saturation
        float excess = hum[id] - sat;
        if (excess > 0.0f) {
            float cond = 0.80f * excess;  // stronger condensation
            hum[id]   -= cond;
            cloud[id] += cond;
        }
        cloud[id] = fmaxf(0.0f, cloud[id] - 0.01f * dt); // slower fallout
    }
}

static void SimStep(float dt){
    AddInflow(U, H);
    AddGroundEvaporation(H, dt);
    AddBuoyancy(V, T, H, dt);

    for(int id=0; id<NX*NY*NZ; ++id){ U0[id]=U[id]; V0[id]=V[id]; W0[id]=W[id]; }
    Diffuse(U,U0,g_visc,dt,6,1);
    Diffuse(V,V0,g_visc,dt,6,1);
    Diffuse(W,W0,g_visc,dt,6,1);
    EnforceSolids(U,V,W);

    AdvectVelocityField(U0,V0,W0, U,V,W, dt);
    for(int id=0; id<NX*NY*NZ; ++id){ U[id]=U0[id]; V[id]=V0[id]; W[id]=W0[id]; }
    EnforceSolids(U,V,W);

    ComputeDivergence(U,V,W);
    SolvePressure();
    SubtractPressureGradient(U,V,W);

    for(int id=0; id<NX*NY*NZ; ++id){ T0[id]=T[id]; H0[id]=H[id]; C0[id]=C[id]; }
    Diffuse(T,T0,g_diff,dt,4,0);
    Diffuse(H,H0,g_diff,dt,4,0);
    Diffuse(C,C0,g_diff*0.2f,dt,2,0);

    AdvectScalarField(T0,T,U,V,W,dt);
    AdvectScalarField(H0,H,U,V,W,dt);
    AdvectScalarField(C0,C,U,V,W,dt);
    for(int id=0; id<NX*NY*NZ; ++id){ T[id]=T0[id]; H[id]=H0[id]; C[id]=C0[id]; }

    Condense(T,H,C,dt);
}

// ---------- Color maps ----------
static Color TempToColor(float t, float a){
    t=clampf(t,0,1); float r,g,b;
    if(t<0.25f){ float u=t/0.25f; r=0; g=u; b=1; }
    else if(t<0.5f){ float u=(t-0.25f)/0.25f; r=0; g=1; b=1-u; }
    else if(t<0.75f){ float u=(t-0.5f)/0.25f; r=u; g=1; b=0; }
    else { float u=(t-0.75f)/0.25f; r=1; g=1-u; b=0; }
    return (Color){(unsigned char)(r*255),(unsigned char)(g*255),(unsigned char)(b*255),(unsigned char)a};
}
static Color HumToColor(float h, float a){
    h=clampf(h,0,1);
    float r=0.2f*h, g=0.45f+0.4f*h, b=0.8f+0.2f*h;
    return (Color){(unsigned char)(r*255),(unsigned char)(g*255),(unsigned char)(b*255),(unsigned char)a};
}
static Color CloudToColor(float c){
    c=clampf(c,0,1); unsigned char a=(unsigned char)clampf(c*255.0f, 20.0f, 235.0f);
    return (Color){255,255,255,a};
}

// ---------- Slice textures ----------
#define MAX_SLICES 64  // >= max(NX,NY,NZ). Here max=48, so 64 is safe.

typedef enum { AXIS_X=0, AXIS_Y=1, AXIS_Z=2 } SliceAxis;
static Texture2D gSlices[MAX_SLICES];
static int gSliceCount = 0;
static int gSliceW = 0, gSliceH = 0;
static SliceAxis gAxis = AXIS_Y;
static bool gSlicesInit = false;

static void UnloadSlices(void){
    if (!gSlicesInit) return;
    for (int s=0; s<gSliceCount; ++s) if (gSlices[s].id != 0) UnloadTexture(gSlices[s]);
    gSlicesInit = false; gSliceCount=0; gSliceW=gSliceH=0;
}

static void CreateSlices(SliceAxis axis){
    UnloadSlices();
    gAxis = axis;
    if (axis == AXIS_Y) { gSliceCount=NY; gSliceW=NX; gSliceH=NZ; }
    else if (axis == AXIS_X){ gSliceCount=NX; gSliceW=NZ; gSliceH=NY; }
    else { gSliceCount=NZ; gSliceW=NX; gSliceH=NY; }

    Image img = GenImageColor(gSliceW, gSliceH, BLANK);
    for (int s=0; s<gSliceCount; ++s) {
        gSlices[s] = LoadTextureFromImage(img);
        SetTextureFilter(gSlices[s], TEXTURE_FILTER_BILINEAR);
        SetTextureWrap(gSlices[s], TEXTURE_WRAP_CLAMP);
    }
    UnloadImage(img);
    gSlicesInit = true;
}

static void UpdateSlicesTextures(void){
    if (!gSlicesInit) return;

    static Color *pixels = NULL;
    static int capW = 0, capH = 0;
    if (capW != gSliceW || capH != gSliceH || pixels == NULL) {
        if (pixels) MemFree(pixels);
        pixels = MemAlloc(gSliceW * gSliceH * sizeof(Color));
        capW = gSliceW; capH = gSliceH;
    }

    for (int s=0; s<gSliceCount; ++s) {
        if (gAxis == AXIS_Y) {
            int j = s;
            for (int v=0; v<gSliceH; ++v) {
                int k = v;
                for (int u=0; u<gSliceW; ++u) {
                    int i = u;
                    int id = idx3(i,j,k);
                    Color c = (Color){0,0,0,0};
                    if (!Solid[id]) {
                        if (g_mode == 1)      c = TempToColor(T[id], clampf(H[id]*230.0f + 20.0f, 0.0f, 235.0f));
                        else if (g_mode == 2) c = HumToColor(H[id],  clampf(H[id]*255.0f,          10.0f, 235.0f));
                        else                   c = CloudToColor(C[id]);
                    }
                    pixels[v*gSliceW + u] = c;
                }
            }
        } else if (gAxis == AXIS_X) {
            int i = s;
            for (int v=0; v<gSliceH; ++v) {
                int j = v;
                for (int u=0; u<gSliceW; ++u) {
                    int k = u;
                    int id = idx3(i,j,k);
                    Color c = (Color){0,0,0,0};
                    if (!Solid[id]) {
                        if (g_mode == 1)      c = TempToColor(T[id], clampf(H[id]*230.0f + 20.0f, 0.0f, 235.0f));
                        else if (g_mode == 2) c = HumToColor(H[id],  clampf(H[id]*255.0f,          10.0f, 235.0f));
                        else                   c = CloudToColor(C[id]);
                    }
                    pixels[v*gSliceW + u] = c;
                }
            }
        } else { // AXIS_Z
            int k = s;
            for (int v=0; v<gSliceH; ++v) {
                int j = v;
                for (int u=0; u<gSliceW; ++u) {
                    int i = u;
                    int id = idx3(i,j,k);
                    Color c = (Color){0,0,0,0};
                    if (!Solid[id]) {
                        if (g_mode == 1)      c = TempToColor(T[id], clampf(H[id]*230.0f + 20.0f, 0.0f, 235.0f));
                        else if (g_mode == 2) c = HumToColor(H[id],  clampf(H[id]*255.0f,          10.0f, 235.0f));
                        else                   c = CloudToColor(C[id]);
                    }
                    pixels[v*gSliceW + u] = c;
                }
            }
        }
        UpdateTexture(gSlices[s], pixels);
    }
}

// Draw a textured axis-aligned quad in 3D (double-sided)
static void DrawSliceQuad(Texture2D tex, SliceAxis axis, int sIndex){
    rlDisableBackfaceCulling();
    rlSetTexture(tex.id);
    rlBegin(RL_TRIANGLES);
    rlColor4ub(255,255,255,255);

    if (axis == AXIS_Y) {
        float y = (sIndex + 0.5f)*DY;
        rlTexCoord2f(0.0f, 1.0f); rlVertex3f(0.0f, y, 0.0f);
        rlTexCoord2f(1.0f, 1.0f); rlVertex3f(TERRAIN_SIZE_X, y, 0.0f);
        rlTexCoord2f(1.0f, 0.0f); rlVertex3f(TERRAIN_SIZE_X, y, TERRAIN_SIZE_Z);

        rlTexCoord2f(0.0f, 1.0f); rlVertex3f(0.0f, y, 0.0f);
        rlTexCoord2f(1.0f, 0.0f); rlVertex3f(TERRAIN_SIZE_X, y, TERRAIN_SIZE_Z);
        rlTexCoord2f(0.0f, 0.0f); rlVertex3f(0.0f, y, TERRAIN_SIZE_Z);

    } else if (axis == AXIS_X) {
        float x = (sIndex + 0.5f)*DX;
        rlTexCoord2f(0.0f, 1.0f); rlVertex3f(x, 0.0f, 0.0f);
        rlTexCoord2f(1.0f, 1.0f); rlVertex3f(x, 0.0f, TERRAIN_SIZE_Z);
        rlTexCoord2f(1.0f, 0.0f); rlVertex3f(x, WORLD_TOP_Y, TERRAIN_SIZE_Z);

        rlTexCoord2f(0.0f, 1.0f); rlVertex3f(x, 0.0f, 0.0f);
        rlTexCoord2f(1.0f, 0.0f); rlVertex3f(x, WORLD_TOP_Y, TERRAIN_SIZE_Z);
        rlTexCoord2f(0.0f, 0.0f); rlVertex3f(x, WORLD_TOP_Y, 0.0f);

    } else { // AXIS_Z
        float z = (sIndex + 0.5f)*DZ;
        rlTexCoord2f(0.0f, 1.0f); rlVertex3f(0.0f, 0.0f, z);
        rlTexCoord2f(1.0f, 1.0f); rlVertex3f(TERRAIN_SIZE_X, 0.0f, z);
        rlTexCoord2f(1.0f, 0.0f); rlVertex3f(TERRAIN_SIZE_X, WORLD_TOP_Y, z);

        rlTexCoord2f(0.0f, 1.0f); rlVertex3f(0.0f, 0.0f, z);
        rlTexCoord2f(1.0f, 0.0f); rlVertex3f(TERRAIN_SIZE_X, WORLD_TOP_Y, z);
        rlTexCoord2f(0.0f, 0.0f); rlVertex3f(0.0f, WORLD_TOP_Y, z);
    }
    rlEnd();
    rlSetTexture(0);
    rlEnableBackfaceCulling();
}

// ---------- Debug cubes ----------
static void DrawDebugCubes(void){
    BeginBlendMode(BLEND_ALPHA);
    int step = g_drawStep;
    for (int j=0;j<NY;j+=step){
        float y=(j+0.5f)*DY;
        for(int k=0;k<NZ;k+=step){
            float z=(k+0.5f)*DZ;
            for(int i=0;i<NX;i+=step){
                float x=(i+0.5f)*DX;
                int id=idx3(i,j,k);
                if(Solid[id]) continue;
                Color col; bool draw=false;
                if(g_mode==1){
                    float a=clampf(H[id]*220.0f+15.0f,0.0f,235.0f);
                    col=TempToColor(T[id], a);
                    draw = (H[id]>0.03f) || (T[id]<0.3f || T[id]>0.7f);
                } else if(g_mode==2){
                    float a=clampf(H[id]*255.0f, 10.0f, 235.0f);
                    col=HumToColor(H[id], a);
                    draw = (H[id] > 0.08f);
                } else {
                    col=CloudToColor(C[id]);
                    draw=(C[id]>0.01f);
                }
                if(!draw) continue;
                Vector3 pos={x,y,z}, size={DX*0.9f*step, DY*0.85f*step, DZ*0.9f*step};
                DrawCubeV(pos,size,col);
            }
        }
    }
    EndBlendMode();
}

// ---------- Min/Max readout ----------
static void GetFieldMinMax(float* arr, float* outMin, float* outMax){
    float mn=1e9f, mx=-1e9f;
    for(int id=0; id<NX*NY*NZ; ++id){
        if(Solid[id]) continue;
        float v=arr[id];
        if(v<mn) mn=v;
        if(v>mx) mx=v;
    }
    *outMin = (mn==1e9f)?0.0f:mn; *outMax = (mx==-1e9f)?0.0f:mx;
}

// ---------- Axis selection with hysteresis ----------
static SliceAxis ChooseAxisWithHysteresis(Vector3 fwd, SliceAxis current) {
    float ax = fabsf(fwd.x), ay = fabsf(fwd.y), az = fabsf(fwd.z);
    SliceAxis bestA = AXIS_X;
    float best = ax, second = fmaxf(ay, az);
    if (ay > best) { second = fmaxf(ax, az); best = ay; bestA = AXIS_Y; }
    if (az > best) { second = fmaxf(ax, ay); best = az; bestA = AXIS_Z; }
    const float HYST = 0.08f;
    if (best - second < HYST) return current;
    return bestA;
}

// ---------- Main ----------
int main(int argc, char** argv){
    const char* objPath = (argc>1)? argv[1] : "eroded_terrain.obj";
    if(!LoadTerrainHeightFromOBJ(objPath)){
        TraceLog(LOG_ERROR,"Failed to read terrain from %s", objPath);
        return 1;
    }

    InitWindow(1400, 900, "Volumetric Weather (Slices)");
    SetTargetFPS(60);

    Camera3D cam = {0};
    cam.position = (Vector3){140.0f, 130.0f, 360.0f};
    cam.target   = (Vector3){TERRAIN_SIZE_X*0.5f, 30.0f, TERRAIN_SIZE_Z*0.5f};
    cam.up       = (Vector3){0,1,0};
    cam.fovy     = 45.0f;
    cam.projection = CAMERA_PERSPECTIVE;

    Model terrain = BuildTerrainModelFromHeight();

    AllocateFields();
    ResetFields();

    // Initial slice axis (top-down → use Y)
    CreateSlices(AXIS_Y);

    float rot=0.0f, dist=340.0f;
    bool paused=false;

    while(!WindowShouldClose()){
        // Input
        if(IsKeyPressed(KEY_SPACE)) paused=!paused;
        if(IsKeyPressed(KEY_ONE)) g_mode=1;
        if(IsKeyPressed(KEY_TWO)) g_mode=2;
        if(IsKeyPressed(KEY_THREE)) g_mode=3;
        if(IsKeyPressed(KEY_W)) g_wireframeTerrain=!g_wireframeTerrain;
        if(IsKeyPressed(KEY_R)) ResetFields();
        if(IsKeyPressed(KEY_B)) g_debugCubes=!g_debugCubes;
        if(IsKeyPressed(KEY_LEFT_BRACKET))  g_drawStep = (g_drawStep>1)? g_drawStep-1 : 1;
        if(IsKeyPressed(KEY_RIGHT_BRACKET)) g_drawStep = (g_drawStep<8)? g_drawStep+1 : 8;
        if(IsKeyPressed(KEY_COMMA))  g_dt=fmaxf(0.02f, g_dt*0.8f);
        if(IsKeyPressed(KEY_PERIOD)) g_dt=fminf(0.25f, g_dt*1.25f);

        if(IsKeyDown(KEY_LEFT))  rot += 0.01f;
        if(IsKeyDown(KEY_RIGHT)) rot -= 0.01f;
        dist -= GetMouseWheelMove()*8.0f;
        dist = clampf(dist, 120.0f, 900.0f);
        float cx=sinf(rot)*dist, cz=cosf(rot)*dist, cy=dist*0.45f;
        cam.position = Vector3Add(cam.target, (Vector3){cx,cy,cz});

        // Choose slice axis with hysteresis
        Vector3 fwd = Vector3Normalize(Vector3Subtract(cam.target, cam.position));
        SliceAxis desiredAxis = ChooseAxisWithHysteresis(fwd, gAxis);
        if (desiredAxis != gAxis) CreateSlices(desiredAxis);

        // Step sim
        if(!paused){ for(int s=0; s<g_substeps; ++s) SimStep(g_dt); }

        // Update slice textures from current field
        UpdateSlicesTextures();

        // Draw
        BeginDrawing();
        ClearBackground((Color){10,10,14,255});
        BeginMode3D(cam);

            // Terrain
            if (g_wireframeTerrain) DrawModelWires(terrain, (Vector3){0}, 1.0f, (Color){200,200,200,255});
            else                    DrawModel(terrain, (Vector3){0}, 1.0f, WHITE);

            // Slices (disable depth writes to avoid white-sheet artifact)
            rlDisableDepthMask();
            BeginBlendMode(BLEND_ALPHA);

            // Back-to-front order along the slicing axis based on view direction sign
            if (gAxis == AXIS_Y) {
                int start = (fwd.y < 0.0f) ? (NY-1) : 0;
                int end   = (fwd.y < 0.0f) ? -1 : NY;
                int step  = (fwd.y < 0.0f) ? -1 : 1;
                for (int j=start; j!=end; j+=step) DrawSliceQuad(gSlices[j], AXIS_Y, j);
            } else if (gAxis == AXIS_X) {
                int start = (fwd.x < 0.0f) ? (NX-1) : 0;
                int end   = (fwd.x < 0.0f) ? -1 : NX;
                int step  = (fwd.x < 0.0f) ? -1 : 1;
                for (int i=start; i!=end; i+=step) DrawSliceQuad(gSlices[i], AXIS_X, i);
            } else { // AXIS_Z
                int start = (fwd.z < 0.0f) ? (NZ-1) : 0;
                int end   = (fwd.z < 0.0f) ? -1 : NZ;
                int step  = (fwd.z < 0.0f) ? -1 : 1;
                for (int k=start; k!=end; k+=step) DrawSliceQuad(gSlices[k], AXIS_Z, k);
            }

            EndBlendMode();
            rlEnableDepthMask();

            // Optional cube debug overlay
            if (g_debugCubes) DrawDebugCubes();

        EndMode3D();

        // Min/max HUD
        float tMin,tMax,hMin,hMax,cMin,cMax;
        GetFieldMinMax(T,&tMin,&tMax);
        GetFieldMinMax(H,&hMin,&hMax);
        GetFieldMinMax(C,&cMin,&cMax);

        DrawFPS(10,10);
        DrawText(paused ? "PAUSED" : "RUNNING", 10, 34, 18, paused?RED:GREEN);
        DrawText(TextFormat("Mode: %s   dt: %.3f   Axis: %s   Cubes:%s",
                            (g_mode==1)?"Temp":(g_mode==2)?"Humidity":"Clouds",
                            g_dt, (gAxis==AXIS_Y)?"Y":(gAxis==AXIS_X)?"X":"Z",
                            g_debugCubes?"ON":"OFF"),
                 10, 56, 18, RAYWHITE);
        DrawText(TextFormat("T[min,max]=[%.2f, %.2f]  H[min,max]=[%.2f, %.2f]  C[min,max]=[%.2f, %.2f]",
                            tMin,tMax,hMin,hMax,cMin,cMax),
                 10, 78, 18, (Color){200,200,200,255});
        DrawText("SPACE pause | 1 Temp | 2 Hum | 3 Clouds | B cubes | [ ] density | , . dt | W wires | R reset",
                 10, 100, 16, (Color){180,180,180,255});

        EndDrawing();
    }

    UnloadSlices();
    UnloadModel(terrain);
    FreeFields();
    CloseWindow();
    return 0;
}
