// obj.c — Photoreal-feel procedural terrain (no textures, no gloss)
// Build (macOS): clang obj.c -o obj -I/path/to/raylib -L/path/to/raylib -lraylib \
//   -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo

#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HASH_SIZE 262144
#define INITIAL_CAPACITY 500000

typedef struct { Vector3 position, normal; Color color; } VertexData;
typedef struct { int vIdx, nIdx, index, occupied; } VNHashEntry;

static VNHashEntry *hashTable=NULL;
static VertexData  *vertices=NULL;
static int *idxBuf=NULL;
static int vertexCount=0, indexCount=0;

static int hashKey(int v,int n){ return ((v*31+n)&(HASH_SIZE-1)); }

static int FindOrAddVertex(int vIdx,int nIdx,Vector3 *tempVerts,Vector3 *tempNorms) {
    int h=hashKey(vIdx,nIdx);
    while(hashTable[h].occupied){
        if(hashTable[h].vIdx==vIdx && hashTable[h].nIdx==nIdx) return hashTable[h].index;
        h=(h+1)&(HASH_SIZE-1);
    }
    Vector3 norm = Vector3Normalize(tempNorms[nIdx]);
    vertices[vertexCount] = (VertexData){
        .position=(Vector3){ tempVerts[vIdx].x, tempVerts[vIdx].y*1.5f, tempVerts[vIdx].z },
        .normal=norm, .color=(Color){80,120,70,255}
    };
    hashTable[h]=(VNHashEntry){vIdx,nIdx,vertexCount,1};
    return vertexCount++;
}

static Model LoadOBJAsModel(const char *filename){
    FILE *f=fopen(filename,"r");
    if(!f){ TraceLog(LOG_ERROR,"Failed to open OBJ: %s",filename); return (Model){0}; }

    Vector3 *tempVerts=malloc(INITIAL_CAPACITY*sizeof(Vector3));
    Vector3 *tempNorms=malloc(INITIAL_CAPACITY*sizeof(Vector3));
    vertices=malloc(INITIAL_CAPACITY*sizeof(VertexData));
    idxBuf=malloc(INITIAL_CAPACITY*sizeof(int));
    hashTable=calloc(HASH_SIZE,sizeof(VNHashEntry));

    int vCount=0,nCount=0; char line[256];
    while(fgets(line,sizeof(line),f)){
        if(line[0]=='v' && line[1]==' '){ float x,y,z; sscanf(line,"v %f %f %f",&x,&y,&z); tempVerts[vCount++] = (Vector3){x,y,z}; }
        else if(line[0]=='v' && line[1]=='n'){ float x,y,z; sscanf(line,"vn %f %f %f",&x,&y,&z); tempNorms[nCount++] = (Vector3){x,y,z}; }
    }
    rewind(f);
    vertexCount=0; indexCount=0;
    while(fgets(line,sizeof(line),f)){
        if(line[0]=='f'){
            int vi[3],ni[3];
            if(sscanf(line,"f %d//%d %d//%d %d//%d",&vi[0],&ni[0],&vi[1],&ni[1],&vi[2],&ni[2])==6){
                for(int i=0;i<3;i++){
                    int v=vi[i]-1, n=ni[i]-1;
                    if(v<0||n<0||v>=vCount||n>=nCount) continue;
                    idxBuf[indexCount++]=FindOrAddVertex(v,n,tempVerts,tempNorms);
                }
            }
        }
    }
    fclose(f); free(tempVerts); free(tempNorms);

    Mesh mesh={0};
    mesh.vertexCount=vertexCount;
    mesh.triangleCount=indexCount/3;
    mesh.vertices=(float*)MemAlloc(vertexCount*3*sizeof(float));
    mesh.normals =(float*)MemAlloc(vertexCount*3*sizeof(float));
    mesh.colors  =(unsigned char*)MemAlloc(vertexCount*4*sizeof(unsigned char));
    mesh.indices =(unsigned short*)MemAlloc(indexCount*sizeof(unsigned short));

    for(int i=0;i<vertexCount;i++){
        mesh.vertices[i*3+0]=vertices[i].position.x;
        mesh.vertices[i*3+1]=vertices[i].position.y;
        mesh.vertices[i*3+2]=vertices[i].position.z;
        mesh.normals [i*3+0]=vertices[i].normal.x;
        mesh.normals [i*3+1]=vertices[i].normal.y;
        mesh.normals [i*3+2]=vertices[i].normal.z;
        mesh.colors  [i*4+0]=vertices[i].color.r;
        mesh.colors  [i*4+1]=vertices[i].color.g;
        mesh.colors  [i*4+2]=vertices[i].color.b;
        mesh.colors  [i*4+3]=255;
    }
    for(int i=0;i<indexCount;i++) mesh.indices[i]=idxBuf[i];
    UploadMesh(&mesh,true);
    return LoadModelFromMesh(mesh);
}

// ====== GLSL: No gloss, Oren–Nayar, procedural normals, strata, AO-ish, aerial perspective ======
static const char *vs_src =
"#version 330\n"
"layout(location=0) in vec3 vertexPosition;\n"
"layout(location=2) in vec3 vertexNormal;\n"
"uniform mat4 mvp; uniform mat4 matModel;\n"
"out vec3 vWorldPos; out vec3 vNormal;\n"
"void main(){ vec4 wp=matModel*vec4(vertexPosition,1.0); vWorldPos=wp.xyz; vNormal=normalize(mat3(matModel)*vertexNormal); gl_Position=mvp*vec4(vertexPosition,1.0); }\n";

static const char *fs_src =
"#version 330\n"
"in vec3 vWorldPos; in vec3 vNormal; out vec4 FragColor;\n"
"uniform vec3 uLightDir; uniform vec3 uCamPos;\n"
"uniform float uScale; uniform float uSnowHeight; uniform float uSnowBlend;\n"
"uniform float uFogDensity; uniform float uNormalAmp; uniform float uAOIntensity;\n"
"uniform float uStrataScale; uniform float uStrataContrast; // rock layering\n"
"uniform vec3 uRockTint; uniform vec3 uGrassTint; uniform vec3 uSnowTint;\n"
"\n"
// hash/noise/fbm
"float hash(vec3 p){ return fract(sin(dot(p,vec3(127.1,311.7,74.7)))*43758.5453); }\n"
"float noise(vec3 p){ vec3 i=floor(p), f=fract(p); vec3 u=f*f*(3.0-2.0*f);\n"
" float n000=hash(i+vec3(0,0,0)), n100=hash(i+vec3(1,0,0));\n"
" float n010=hash(i+vec3(0,1,0)), n110=hash(i+vec3(1,1,0));\n"
" float n001=hash(i+vec3(0,0,1)), n101=hash(i+vec3(1,0,1));\n"
" float n011=hash(i+vec3(0,1,1)), n111=hash(i+vec3(1,1,1));\n"
" float nx00=mix(n000,n100,u.x), nx10=mix(n010,n110,u.x);\n"
" float nx01=mix(n001,n101,u.x), nx11=mix(n011,n111,u.x);\n"
" float nxy0=mix(nx00,nx10,u.y), nxy1=mix(nx01,nx11,u.y);\n"
" return mix(nxy0,nxy1,u.z);\n"
"}\n"
"float fbm(vec3 p){ float a=0.5, s=0.0; vec3 q=p; for(int i=0;i<6;i++){ s+=a*noise(q); q*=2.02; a*=0.5; } return s; }\n"
"\n"
// tri-planar sampling returning mid & fine detail channels
"vec2 triDetail(vec3 wp, vec3 n, float scale){ vec3 aw=pow(abs(n), vec3(4.0)); aw/= (aw.x+aw.y+aw.z+1e-5);\n"
" vec2 acc=vec2(0.0);\n"
" vec3 tx=vec3(wp.y, wp.z, 0.0)/scale; vec3 ty=vec3(wp.x, wp.z, 0.0)/scale; vec3 tz=vec3(wp.x, wp.y, 0.0)/scale;\n"
" float rx=fbm(vec3(tx.xy,0.0)); float ry=fbm(vec3(ty.xy,0.0)); float rz=fbm(vec3(tz.xy,0.0));\n"
" float mid=aw.x*rx + aw.y*ry + aw.z*rz; float fine=fbm(wp/(scale*0.5));\n"
" return vec2(mid,fine);\n"
"}\n"
"\n// gradient of fbm for normal perturbation (finite differences)\n"
"vec3 fbmGrad(vec3 p){ float e=0.25; float c=fbm(p); return vec3(fbm(p+vec3(e,0,0))-c, fbm(p+vec3(0,e,0))-c, fbm(p+vec3(0,0,e))-c)/e; }\n"
"\n// Oren–Nayar diffuse (rough rocky scatter)\n"
"float orenNayar(vec3 N, vec3 L, vec3 V, float sigma){ float NdL=max(dot(N,L),0.0); float NdV=max(dot(N,V),0.0);\n"
" float s2=sigma*sigma; float A=1.0 - (s2/(2.0*(s2+0.33))); float B=0.45*s2/(s2+0.09);\n"
" float alpha=max(acos(NdL), acos(NdV)); float beta=min(acos(NdL), acos(NdV));\n"
" float LV = max(0.0, dot(normalize(V - N*NdV), normalize(L - N*NdL)));\n"
" return NdL * (A + B*LV*sin(alpha)*tan(beta)); }\n"
"\n// simple cavity/ambient occlusion proxy from noise & slope (not SSAO, just believable)\n"
"float cavityAO(vec3 wp, vec3 N, float intensity){ float c = fbm(wp*0.08); float slope=1.0-clamp(dot(N,vec3(0,1,0)),0.0,1.0);\n"
" float concave = smoothstep(0.45,0.75,c) * 0.7 + slope*0.2; return clamp(1.0 - concave*intensity, 0.0, 1.0); }\n"
"\n// sedimentary strata bands along a geologic direction\n"
"float strata(vec3 wp, float scale, float contrast){ float s = sin(dot(wp, normalize(vec3(0.2,1.0,0.05)))*scale);\n"
" s = smoothstep(-0.2,0.2,s); return mix(1.0-contrast, 1.0+contrast, s); }\n"
"\n// tone map (ACES-ish) + gamma\n"
"vec3 aces(vec3 x){ const float a=2.51,b=0.03,c=2.43,d=0.59,e=0.14; return clamp((x*(a*x+b))/(x*(c*x+d)+e),0.0,1.0); }\n"
"\nvoid main(){\n"
"  vec3 N = normalize(vNormal);\n"
"  vec3 L = normalize(uLightDir);\n"
"  vec3 V = normalize(uCamPos - vWorldPos);\n"
"\n"
"  // micro normal from fbm gradient; project onto tangent plane so we don't shrink N\n"
"  vec3 g = fbmGrad(vWorldPos*0.15);\n"
"  g = normalize(g*2.0-1.0);\n"
"  vec3 tPerturb = normalize(g - N*dot(g,N));\n"
"  N = normalize(N + tPerturb*uNormalAmp);\n"
"\n"
"  // material layers\n"
"  float slope = 1.0 - clamp(dot(N,vec3(0,1,0)),0.0,1.0);\n"
"  vec2 d = triDetail(vWorldPos, N, uScale);\n"
"  float mid=d.x, fine=d.y;\n"
"  float rockW = smoothstep(0.35,0.65,slope);\n"
"  float snowH = smoothstep(uSnowHeight - uSnowBlend, uSnowHeight + uSnowBlend, vWorldPos.y);\n"
"  float snowSlope = smoothstep(0.7,0.3,slope);\n"
"  float snowW = snowH * snowSlope;\n"
"\n"
"  float str = strata(vWorldPos, uStrataScale, uStrataContrast);\n"
"  vec3 rock = uRockTint * (0.55 + 0.45*mid) * (0.9 + 0.1*fine) * str;\n"
"  vec3 grass = uGrassTint * (0.60 + 0.40*mid) * (0.92 + 0.08*fine);\n"
"  vec3 albedo = mix(grass, rock, rockW);\n"
"  albedo = mix(albedo, uSnowTint*(0.9 + 0.1*mid), snowW);\n"
"\n"
"  // lighting: hemisphere ambient + Oren–Nayar (no specular)\n"
"  float hemi = 0.5 + 0.5*N.y;\n"
"  vec3 ambient = mix(vec3(0.22,0.24,0.26), vec3(0.62,0.70,0.80), hemi) * 0.35;\n"
"  float diff = orenNayar(N,L,V, 0.7); // sigma in radians, ~0.7 ≈ rough rock\n"
"\n"
"  // cavity/ao\n"
"  float ao = cavityAO(vWorldPos, N, uAOIntensity);\n"
"  vec3 color = albedo * (ambient + diff) * ao;\n"
"\n"
"  // aerial perspective (height-aware fog)\n"
"  float dist = length(uCamPos - vWorldPos);\n"
"  float fog = 1.0 - exp(-uFogDensity * dist * clamp(0.3 + 0.7*clamp((vWorldPos.y/300.0),0.0,1.0), 0.3, 1.0));\n"
"  vec3 fogCol = vec3(0.78,0.86,0.95); // sky haze\n"
"  color = mix(color, fogCol, clamp(fog,0.0,1.0));\n"
"\n"
"  color = aces(color); color = pow(color, vec3(1.0/2.2));\n"
"  FragColor = vec4(color,1.0);\n"
"}\n";

static Shader LoadTerrainShader(void){ return LoadShaderFromMemory(vs_src, fs_src); }

int main(void){
    InitWindow(1200,800,"OBJ — Mountain Look (No Gloss)");
    SetTargetFPS(60);

    const char *objPath="eroded_terrain.obj";
    Model model=LoadOBJAsModel(objPath);
    if(model.meshCount==0){ CloseWindow(); return 1; }

    Shader sh=LoadTerrainShader();
    for(int i=0;i<model.materialCount;i++) model.materials[i].shader=sh;

    int locLightDir   = GetShaderLocation(sh,"uLightDir");
    int locCamPos     = GetShaderLocation(sh,"uCamPos");
    int locScale      = GetShaderLocation(sh,"uScale");
    int locSnowHeight = GetShaderLocation(sh,"uSnowHeight");
    int locSnowBlend  = GetShaderLocation(sh,"uSnowBlend");
    int locFogDensity = GetShaderLocation(sh,"uFogDensity");
    int locNormalAmp  = GetShaderLocation(sh,"uNormalAmp");
    int locAOInt      = GetShaderLocation(sh,"uAOIntensity");
    int locStrataSc   = GetShaderLocation(sh,"uStrataScale");
    int locStrataCt   = GetShaderLocation(sh,"uStrataContrast");
    int locRockTint   = GetShaderLocation(sh,"uRockTint");
    int locGrassTint  = GetShaderLocation(sh,"uGrassTint");
    int locSnowTint   = GetShaderLocation(sh,"uSnowTint");

    Vector3 lightDir = Vector3Normalize((Vector3){0.35f,1.0f,0.45f});
    Vector3 camPos = {0};
    float scaleVal=10.0f;
    float snowY=140.0f, snowBlend=24.0f;
    float fogDens=0.0025f;     // increase for more atmosphere
    float normalAmp=0.35f;     // cragginess
    float aoIntensity=0.7f;    // cavity darkening
    float strataScale=0.08f;   // band frequency
    float strataContrast=0.15f;// band strength
    Vector3 rockTint=(Vector3){0.48f,0.46f,0.44f};
    Vector3 grassTint=(Vector3){0.26f,0.44f,0.24f};
    Vector3 snowTint=(Vector3){0.92f,0.96f,1.0f};

    SetShaderValue(sh,locLightDir,&lightDir,SHADER_UNIFORM_VEC3);
    SetShaderValue(sh,locScale,&scaleVal,SHADER_UNIFORM_FLOAT);
    SetShaderValue(sh,locSnowHeight,&snowY,SHADER_UNIFORM_FLOAT);
    SetShaderValue(sh,locSnowBlend,&snowBlend,SHADER_UNIFORM_FLOAT);
    SetShaderValue(sh,locFogDensity,&fogDens,SHADER_UNIFORM_FLOAT);
    SetShaderValue(sh,locNormalAmp,&normalAmp,SHADER_UNIFORM_FLOAT);
    SetShaderValue(sh,locAOInt,&aoIntensity,SHADER_UNIFORM_FLOAT);
    SetShaderValue(sh,locStrataSc,&strataScale,SHADER_UNIFORM_FLOAT);
    SetShaderValue(sh,locStrataCt,&strataContrast,SHADER_UNIFORM_FLOAT);
    SetShaderValue(sh,locRockTint,&rockTint,SHADER_UNIFORM_VEC3);
    SetShaderValue(sh,locGrassTint,&grassTint,SHADER_UNIFORM_VEC3);
    SetShaderValue(sh,locSnowTint,&snowTint,SHADER_UNIFORM_VEC3);

    bool wireframe=false;
    Camera3D cam={
        .position={128.0f,120.0f,330.0f},
        .target={128.0f,0.0f,128.0f},
        .up={0.0f,1.0f,0.0f},
        .fovy=45.0f, .projection=CAMERA_PERSPECTIVE
    };
    float rotation=0.0f, distance=300.0f;

    while(!WindowShouldClose()){
        if(IsKeyPressed(KEY_W)) wireframe=!wireframe;
        if(IsKeyDown(KEY_LEFT)) rotation+=0.01f;
        if(IsKeyDown(KEY_RIGHT)) rotation-=0.01f;
        distance-=GetMouseWheelMove()*5.0f;
        distance=Clamp(distance,50.0f,800.0f);

        // live tweaks
        if(IsKeyDown(KEY_LEFT_BRACKET)){ scaleVal=fmaxf(3.0f,scaleVal-0.15f); SetShaderValue(sh,locScale,&scaleVal,SHADER_UNIFORM_FLOAT); }
        if(IsKeyDown(KEY_RIGHT_BRACKET)){ scaleVal=fminf(40.0f,scaleVal+0.15f); SetShaderValue(sh,locScale,&scaleVal,SHADER_UNIFORM_FLOAT); }
        if(IsKeyDown(KEY_MINUS)){ normalAmp=Clamp(normalAmp-0.01f,0.0f,1.0f); SetShaderValue(sh,locNormalAmp,&normalAmp,SHADER_UNIFORM_FLOAT); }
        if(IsKeyDown(KEY_EQUAL)){ normalAmp=Clamp(normalAmp+0.01f,0.0f,1.0f); SetShaderValue(sh,locNormalAmp,&normalAmp,SHADER_UNIFORM_FLOAT); }
        if(IsKeyDown(KEY_FIVE)){ snowY-=1.0f; SetShaderValue(sh,locSnowHeight,&snowY,SHADER_UNIFORM_FLOAT); }
        if(IsKeyDown(KEY_SIX)){  snowY+=1.0f; SetShaderValue(sh,locSnowHeight,&snowY,SHADER_UNIFORM_FLOAT); }
        if(IsKeyDown(KEY_SEMICOLON)){ fogDens=fmaxf(0.0f,fogDens-0.0002f); SetShaderValue(sh,locFogDensity,&fogDens,SHADER_UNIFORM_FLOAT); }
        if(IsKeyDown(KEY_APOSTROPHE)){ fogDens=fminf(0.01f,fogDens+0.0002f); SetShaderValue(sh,locFogDensity,&fogDens,SHADER_UNIFORM_FLOAT); }

        float camX=sinf(rotation)*distance, camZ=cosf(rotation)*distance, camY=distance*0.5f;
        cam.position=Vector3Add(cam.target,(Vector3){camX,camY,camZ});
        camPos=cam.position;
        SetShaderValue(sh,locCamPos,&camPos,SHADER_UNIFORM_VEC3);

        BeginDrawing();
            ClearBackground((Color){12,12,12,255});
            BeginMode3D(cam);
                if(wireframe){
                    rlSetLineWidth(1.0f); rlDrawRenderBatchActive(); rlBegin(RL_LINES);
                    for(int i=0;i<indexCount;i+=3){
                        Vector3 v0=vertices[idxBuf[i+0]].position;
                        Vector3 v1=vertices[idxBuf[i+1]].position;
                        Vector3 v2=vertices[idxBuf[i+2]].position;
                        rlColor4ub(255,255,255,255);
                        rlVertex3f(v0.x,v0.y,v0.z); rlVertex3f(v1.x,v1.y,v1.z);
                        rlVertex3f(v1.x,v1.y,v1.z); rlVertex3f(v2.x,v2.y,v2.z);
                        rlVertex3f(v2.x,v2.y,v2.z); rlVertex3f(v0.x,v0.y,v0.z);
                    } rlEnd();
                } else {
                    DrawModel(model,(Vector3){0},1.0f,WHITE);
                }
            EndMode3D();
            DrawFPS(10,10);
            DrawText(TextFormat("OBJ: %s%s", objPath, wireframe?" (Wireframe)":""),
                     10,30,18,GREEN);
            DrawText("[/] detail  -/_ normal  5/6 snow  ;/' fog",10,52,16,RAYWHITE);
        EndDrawing();
    }
    UnloadModel(model); UnloadShader(sh);
    free(vertices); free(idxBuf); free(hashTable);
    CloseWindow(); return 0;
}
