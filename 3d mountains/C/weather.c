// terrain_weather_sim.c (Corrected & Hardened Version)
// Build (Linux/macOS): clang -O2 -std=c99 terrain_weather_sim.c -lraylib -lm -lpthread -ldl -lrt -lX11 -o weather
// macOS (homebrew raylib): clang -O2 -std=c99 terrain_weather_sim.c -lraylib -framework IOKit -framework Cocoa -framework OpenGL -o weather
// Windows (MinGW example): gcc -O2 -std=c99 terrain_weather_sim.c -lraylib -lopengl32 -lgdi32 -lwinmm -o weather.exe
//
// This program loads an OBJ terrain mesh, builds a heightfield, and runs a lightweight layered
// atmospheric simulation (wind, temperature, humidity, clouds, precipitation) for visual
// experimentation. Emphasis: *qualitatively plausible* rather than physically exact.
//
// Controls:
//   Mouse Wheel   : Zoom camera distance
//   Left/Right    : Orbit camera
//   W             : Toggle terrain wireframe
//   1 / 2 / 3     : Select overlay (Temperature / Humidity / Cloud Water)
//   4             : Toggle wind vectors
//   SPACE (hold)  : Inject surface moisture under cursor
//   R             : Reset simulation
//   +/- or = / -  : Adjust time scale
//   ESC           : Quit
//
// Notes:
// * Assumes OBJ X,Z extents define a roughly rectangular area. We normalize to a GRID_W x GRID_H grid.
// * Height values are vertically exaggerated (TERRAIN_VERTICAL_SCALE) consistently in mesh & heightfield.
// * Semi-Lagrangian advection used for stability.
// * All units for horizontal grid assume 1 cell = arbitrary distance (scaled from model bounds). If you
//   want real meters, set CELL_SIZE accordingly and adjust wind speeds.
//
// Recommended next steps (optional):
//   - Introduce separate physical scaling for meters per cell
//   - Add pressure gradient force for dynamic wind evolution
//   - Move update to a fixed timestep accumulator
//   - GPU compute (shader storage buffer or textures) for >512 grids
//
// --------------------------------------------------------------------------------------------

#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>

// ---------------- Configuration ----------------
#define GRID_W 256
#define GRID_H 256
#define LAYERS 3

static const float LAYER_THICKNESS = 200.0f;          // Approx height of each atmospheric slab (m)
static const float TERRAIN_VERTICAL_SCALE = 1.5f;      // Vertical exaggeration applied uniformly

// Simulation parameters (tunable)
static float BASE_SEA_LEVEL_TEMP   = 20.0f;            // °C
static float LAPSE_RATE            = 6.5f / 1000.0f;   // °C per meter
static float CONDENSE_RATE         = 0.5f;             // Fraction of supersaturation condensed per step
static float RAIN_THRESHOLD        = 0.3f;             // Cloud water threshold for precipitation
static float RAIN_RATE             = 0.05f;            // Fraction of excess cloud removed as rain
static float LATENT_HEAT_FACTOR    = 0.1f;             // Heating on condensation
static float EVAP_COOLING          = 0.05f;            // Cooling from precipitation
static float DIURNAL_AMPLITUDE     = 4.0f;             // Surface diurnal temperature swing (°C peak-to-mid)
static float TIME_SCALE            = 600.0f;           // Simulated seconds per real second
static float WIND_BASE_U           = 8.0f;             // Initial west->east base wind (m/s)
static float WIND_BASE_V           = 2.0f;             // Initial south->north base wind (m/s)
static float OROGRAPHIC_SCALE      = 0.01f;            // Scaling for vertical velocity from slope
static float ADIABATIC_COOL_DRY    = 9.8f / 1000.0f;   // Dry adiabatic lapse for vertical motion (°C/m)
static float MOISTURE_INJECT_AMOUNT= 0.2f;             // Added q when injecting moisture

// Visual toggles
static int showTemp  = 1;
static int showHum   = 0;
static int showCloud = 1;
static int showWind  = 1;

// Data structures

typedef struct {
    float height;          // Surface height (m)
    float T[LAYERS];       // Temperature per layer (°C)
    float q[LAYERS];       // Specific humidity-like scalar (arbitrary units)
    float cloud[LAYERS];   // Cloud water mixing ratio
} Cell;

typedef struct {
    float u[LAYERS];       // Wind X (grid units / s)
    float v[LAYERS];       // Wind Z (grid units / s)
    float w[LAYERS];       // Diagnostic vertical velocity (m/s)
} FlowCell;

static Cell      grid[GRID_W * GRID_H];
static FlowCell  flowField[GRID_W * GRID_H];
static float     advectT[LAYERS][GRID_W * GRID_H];
static float     advectQ[LAYERS][GRID_W * GRID_H];
static float     advectC[LAYERS][GRID_W * GRID_H];
static float     heightField[GRID_W * GRID_H];

static inline int IDX(int x, int y) { return y * GRID_W + x; }

// ---------------- Utility Functions ----------------
static float SampleBilinear(float *field, float x, float z) {
    if (x < 0) x = 0; if (z < 0) z = 0;
    if (x > GRID_W - 1) x = GRID_W - 1;
    if (z > GRID_H - 1) z = GRID_H - 1;
    int x0 = (int)x; int z0 = (int)z;
    int x1 = x0 + 1; if (x1 >= GRID_W) x1 = x0;
    int z1 = z0 + 1; if (z1 >= GRID_H) z1 = z0;
    float tx = x - x0; float tz = z - z0;
    float f00 = field[IDX(x0,z0)];
    float f10 = field[IDX(x1,z0)];
    float f01 = field[IDX(x0,z1)];
    float f11 = field[IDX(x1,z1)];
    float a = f00 + (f10 - f00) * tx;
    float b = f01 + (f11 - f01) * tx;
    return a + (b - a) * tz;
}

static float Saturation(float T) { // Tetens approximation proxy
    float es = 6.1078f * expf((17.269f * T) / (T + 237.3f));
    return es; // treat as capacity measure
}

// ---------------- Model / Heightfield Loader ----------------
static Model LoadOBJBuildHeight(const char *filename) {
    Model model = LoadModel(filename);
    if (model.meshCount == 0) {
        TraceLog(LOG_ERROR, "LoadModel -> meshCount=0 (%s). Using fallback plane.", filename);
        Mesh plane = GenMeshPlane((float)(GRID_W - 1), (float)(GRID_H - 1), 2, 2);
        model = LoadModelFromMesh(plane);
    }

    Mesh *mesh = &model.meshes[0];
    if (mesh->vertexCount == 0 || mesh->vertices == NULL) {
        TraceLog(LOG_ERROR, "Primary mesh empty. Flat heightfield.");
        for (int i = 0; i < GRID_W * GRID_H; ++i) heightField[i] = 0.0f;
        return model;
    }

    float minX = FLT_MAX, maxX = -FLT_MAX, minZ = FLT_MAX, maxZ = -FLT_MAX;
    for (int i = 0; i < mesh->vertexCount; ++i) {
        float vx = mesh->vertices[i*3 + 0];
        float vz = mesh->vertices[i*3 + 2];
        if (vx < minX) minX = vx; if (vx > maxX) maxX = vx;
        if (vz < minZ) minZ = vz; if (vz > maxZ) maxZ = vz;
    }
    if (minX >= maxX) maxX = minX + 1.0f;
    if (minZ >= maxZ) maxZ = minZ + 1.0f;
    float invSpanX = 1.0f / (maxX - minX);
    float invSpanZ = 1.0f / (maxZ - minZ);

    for (int i = 0; i < GRID_W * GRID_H; ++i) heightField[i] = -FLT_MAX;

    for (int i = 0; i < mesh->vertexCount; ++i) {
        float vx = mesh->vertices[i*3 + 0];
        float vy = mesh->vertices[i*3 + 1] * TERRAIN_VERTICAL_SCALE;
        float vz = mesh->vertices[i*3 + 2];
        int gx = (int)((vx - minX) * invSpanX * (GRID_W - 1) + 0.5f);
        int gz = (int)((vz - minZ) * invSpanZ * (GRID_H - 1) + 0.5f);
        if (gx >= 0 && gx < GRID_W && gz >= 0 && gz < GRID_H) {
            float *cellH = &heightField[IDX(gx,gz)];
            if (vy > *cellH) *cellH = vy;
        }
    }
    // Fill holes
    for (int iter = 0; iter < 6; ++iter) {
        for (int z = 1; z < GRID_H - 1; ++z) {
            for (int x = 1; x < GRID_W - 1; ++x) {
                int id = IDX(x,z);
                if (heightField[id] < -1e5f) {
                    float sum = 0, cnt = 0;
                    for (int dz = -1; dz <= 1; ++dz)
                        for (int dx = -1; dx <= 1; ++dx) {
                            int nid = IDX(x+dx,z+dz);
                            float h = heightField[nid];
                            if (h > -1e5f) { sum += h; cnt += 1; }
                        }
                    if (cnt > 0) heightField[id] = sum / cnt;
                }
            }
        }
    }
    for (int i = 0; i < GRID_W * GRID_H; ++i) if (heightField[i] < -1e5f) heightField[i] = 0.0f;

    // Apply vertical scale to actual mesh vertices so drawing matches heightfield
    for (int i = 0; i < mesh->vertexCount; ++i) mesh->vertices[i*3 + 1] *= TERRAIN_VERTICAL_SCALE;
    UploadMesh(mesh, false);

    TraceLog(LOG_INFO, "Heightfield built: X:[%.2f, %.2f] Z:[%.2f, %.2f]", minX, maxX, minZ, maxZ);
    return model;
}

// ---------------- Simulation Initialization ----------------
static void Weather_Reset(void) {
    for (int z = 0; z < GRID_H; ++z) {
        for (int x = 0; x < GRID_W; ++x) {
            int i = IDX(x,z);
            float h = heightField[i];
            grid[i].height = h;
            for (int L = 0; L < LAYERS; ++L) {
                float layerAlt = h + (L + 0.5f) * LAYER_THICKNESS;
                float baseT = BASE_SEA_LEVEL_TEMP - LAPSE_RATE * layerAlt;
                grid[i].T[L] = baseT;
                grid[i].q[L] = 0.6f - 0.1f * (float)L * 0.5f; // Slightly drier aloft
                if (grid[i].q[L] < 0.1f) grid[i].q[L] = 0.1f;
                grid[i].cloud[L] = 0.0f;
                flowField[i].u[L] = WIND_BASE_U * (1.0f - 0.1f * L);
                flowField[i].v[L] = WIND_BASE_V * (1.0f - 0.1f * L);
                flowField[i].w[L] = 0.0f;
            }
        }
    }
}

// ---------------- Simulation Core ----------------
static void Weather_ComputeVertical(void) {
    for (int z = 1; z < GRID_H - 1; ++z) {
        for (int x = 1; x < GRID_W - 1; ++x) {
            int i = IDX(x,z);
            float hx1 = heightField[IDX(x+1,z)];
            float hx0 = heightField[IDX(x-1,z)];
            float hz1 = heightField[IDX(x,z+1)];
            float hz0 = heightField[IDX(x,z-1)];
            float slopeX = (hx1 - hx0) * 0.5f;   // d(height)/dx
            float slopeZ = (hz1 - hz0) * 0.5f;   // d(height)/dz
            for (int L = 0; L < LAYERS; ++L) {
                float u = flowField[i].u[L];
                float v = flowField[i].v[L];
                flowField[i].w[L] = (u * slopeX + v * slopeZ) * OROGRAPHIC_SCALE; // Exaggerated vertical
            }
        }
    }
}

static void Weather_Advection(float dt) {
    for (int L = 0; L < LAYERS; ++L) {
        for (int i = 0; i < GRID_W * GRID_H; ++i) {
            advectT[L][i] = grid[i].T[L];
            advectQ[L][i] = grid[i].q[L];
            advectC[L][i] = grid[i].cloud[L];
        }
    }
    for (int z = 0; z < GRID_H; ++z) {
        for (int x = 0; x < GRID_W; ++x) {
            int i = IDX(x,z);
            for (int L = 0; L < LAYERS; ++L) {
                float u = flowField[i].u[L];
                float v = flowField[i].v[L];
                float x_prev = x - u * dt; // cell size = 1
                float z_prev = z - v * dt;
                grid[i].T[L]     = SampleBilinear(advectT[L], x_prev, z_prev);
                grid[i].q[L]     = SampleBilinear(advectQ[L], x_prev, z_prev);
                grid[i].cloud[L] = SampleBilinear(advectC[L], x_prev, z_prev);
            }
        }
    }
}

static void Weather_LocalPhysics(float dt, float simTime) {
    float dayFrac = fmodf(simTime / 86400.0f, 1.0f); // 0..1
    float diurnal = sinf(dayFrac * 2.0f * PI);       // -1..1
    for (int i = 0; i < GRID_W * GRID_H; ++i) {
        for (int L = 0; L < LAYERS; ++L) {
            float *T = &grid[i].T[L];
            float *q = &grid[i].q[L];
            float *c = &grid[i].cloud[L];
            float w  = flowField[i].w[L];
            *T -= w * dt * ADIABATIC_COOL_DRY; // vertical motion cooling/heating
            float layerFactor = 1.0f - 0.3f * L; // weaker diurnal higher up
            *T += diurnal * (DIURNAL_AMPLITUDE * layerFactor) * dt * 0.001f; // scaled small per step
            float qs = Saturation(*T);
            if (*q > qs) { // condensation
                float excess = (*q - qs) * CONDENSE_RATE;
                *q -= excess; *c += excess; *T += LATENT_HEAT_FACTOR * excess;
            }
            if (*c > RAIN_THRESHOLD) { // precipitation
                float rain = (*c - RAIN_THRESHOLD) * RAIN_RATE;
                *c -= rain; *q -= rain * 0.2f; *T -= EVAP_COOLING * rain;
                if (*q < 0) *q = 0;
            }
        }
    }
}

static void Weather_Step(float dtSim, float simTime) {
    Weather_ComputeVertical();
    Weather_Advection(dtSim);
    Weather_LocalPhysics(dtSim, simTime);
}

static void Weather_InjectMoisture(int gx, int gz, float radius) {
    for (int z = gz - (int)radius; z <= gz + (int)radius; ++z) {
        if (z < 0 || z >= GRID_H) continue;
        for (int x = gx - (int)radius; x <= gx + (int)radius; ++x) {
            if (x < 0 || x >= GRID_W) continue;
            float dx = x - gx; float dz = z - gz; float d2 = dx*dx + dz*dz; float r2 = radius*radius;
            if (d2 <= r2) {
                int i = IDX(x,z);
                float factor = 1.0f - d2 / r2;
                grid[i].q[0] += MOISTURE_INJECT_AMOUNT * factor;
                if (grid[i].q[0] > 2.0f) grid[i].q[0] = 2.0f;
            }
        }
    }
}

// ---------------- Rendering Helpers ----------------
static Color ColorRamp(float v) {
    // Normalized 0..1 -> blue->cyan->green->yellow->red
    float r,g,b;
    if (v < 0.25f) { float t=v/0.25f; r=0; g=t; b=1; }
    else if (v < 0.5f) { float t=(v-0.25f)/0.25f; r=0; g=1; b=1-t; }
    else if (v < 0.75f) { float t=(v-0.5f)/0.25f; r=t; g=1; b=0; }
    else { float t=(v-0.75f)/0.25f; r=1; g=1-t; b=0; }
    return (Color){(unsigned char)(r*255),(unsigned char)(g*255),(unsigned char)(b*255),120};
}

static void Weather_DrawOverlay(void) {
    rlDisableBackfaceCulling();
    rlDisableDepthMask();
    for (int z = 0; z < GRID_H - 1; ++z) {
        rlBegin(RL_QUADS);
        for (int x = 0; x < GRID_W - 1; ++x) {
            int i00 = IDX(x,z); int i10 = IDX(x+1,z); int i11 = IDX(x+1,z+1); int i01 = IDX(x,z+1);
            float h00 = heightField[i00]; float h10 = heightField[i10]; float h11 = heightField[i11]; float h01 = heightField[i01];
            float s00=0,s10=0,s11=0,s01=0;
            if (showTemp) { s00=grid[i00].T[0]; s10=grid[i10].T[0]; s11=grid[i11].T[0]; s01=grid[i01].T[0]; }
            else if (showHum) { s00=grid[i00].q[0]; s10=grid[i10].q[0]; s11=grid[i11].q[0]; s01=grid[i01].q[0]; }
            else if (showCloud){ s00=grid[i00].cloud[1]; s10=grid[i10].cloud[1]; s11=grid[i11].cloud[1]; s01=grid[i01].cloud[1]; }
            float minV = showTemp ? (BASE_SEA_LEVEL_TEMP - 25.0f) : 0.0f;
            float maxV = showTemp ? (BASE_SEA_LEVEL_TEMP + 15.0f) : 1.5f;
            float n00=(s00-minV)/(maxV-minV); if(n00<0)n00=0; if(n00>1)n00=1;
            float n10=(s10-minV)/(maxV-minV); if(n10<0)n10=0; if(n10>1)n10=1;
            float n11=(s11-minV)/(maxV-minV); if(n11<0)n11=0; if(n11>1)n11=1;
            float n01=(s01-minV)/(maxV-minV); if(n01<0)n01=0; if(n01>1)n01=1;
            Color c00=ColorRamp(n00), c10=ColorRamp(n10), c11=ColorRamp(n11), c01=ColorRamp(n01);
            rlColor4ub(c00.r,c00.g,c00.b,c00.a); rlVertex3f((float)x, h00+0.5f, (float)z);
            rlColor4ub(c10.r,c10.g,c10.b,c10.a); rlVertex3f((float)(x+1), h10+0.5f, (float)z);
            rlColor4ub(c11.r,c11.g,c11.b,c11.a); rlVertex3f((float)(x+1), h11+0.5f, (float)(z+1));
            rlColor4ub(c01.r,c01.g,c01.b,c01.a); rlVertex3f((float)x, h01+0.5f, (float)(z+1));
        }
        rlEnd();
    }
    rlEnableDepthMask();
    rlEnableBackfaceCulling();
}

static void Weather_DrawWindVectors(void) {
    for (int z = 0; z < GRID_H; z += 8) {
        for (int x = 0; x < GRID_W; x += 8) {
            int i = IDX(x,z);
            float u = flowField[i].u[0]; float v = flowField[i].v[0];
            float len = sqrtf(u*u + v*v);
            if (len < 0.05f) continue;
            Vector3 p = (Vector3){(float)x, heightField[i] + 2.0f, (float)z};
            Vector3 q = (Vector3){p.x + u*0.5f, p.y, p.z + v*0.5f};
            DrawLine3D(p,q, SKYBLUE);
            Vector3 dir = Vector3Normalize((Vector3){u,0,v});
            Vector3 left = (Vector3){-dir.z,0,dir.x};
            Vector3 head1 = Vector3Add(q, Vector3Scale(Vector3Add(dir,left), -0.8f));
            Vector3 head2 = Vector3Add(q, Vector3Scale(Vector3Subtract(dir,left), -0.8f));
            DrawLine3D(q, head1, SKYBLUE); DrawLine3D(q, head2, SKYBLUE);
        }
    }
}

static void Weather_DrawClouds(void) {
    for (int z = 0; z < GRID_H; z += 4) {
        for (int x = 0; x < GRID_W; x += 4) {
            int i = IDX(x,z);
            float c = grid[i].cloud[1];
            if (c > 0.05f) {
                float h = heightField[i] + LAYER_THICKNESS * 1.0f;
                float radius = 2.0f + c * 8.0f;
                unsigned char alpha = (unsigned char)Clamp(c * 255.0f, 40.0f, 220.0f);
                DrawSphereEx((Vector3){(float)x, h + 20.0f, (float)z}, radius, 8, 8, (Color){255,255,255,alpha});
            }
        }
    }
}

// ---------------- Mouse Picking ----------------
static int MouseToGrid(int *outX, int *outZ, Camera3D cam) {
    Vector2 mouse = GetMousePosition();
    Ray ray = GetMouseRay(mouse, cam);
    for (float t = 0; t < 2000.0f; t += 5.0f) {
        Vector3 pos = Vector3Add(ray.position, Vector3Scale(ray.direction, t));
        int gx = (int)roundf(pos.x); int gz = (int)roundf(pos.z);
        if (gx >= 0 && gx < GRID_W && gz >= 0 && gz < GRID_H) {
            float terrainH = heightField[IDX(gx,gz)] + 2.0f;
            if (pos.y <= terrainH) { *outX = gx; *outZ = gz; return 1; }
        }
    }
    return 0;
}

// ---------------- Main ----------------
int main(void) {
    InitWindow(1400, 900, "Terrain Weather Simulation");
    SetTargetFPS(60);

    const char *terrainFile = "eroded_terrain.obj"; // Adjust path as needed
    Model terrain = LoadOBJBuildHeight(terrainFile);

    if (terrain.meshCount == 0 || terrain.meshes[0].vertexCount == 0) {
        TraceLog(LOG_WARNING, "Terrain mesh empty; running on flat ground.");
    }

    for (int i = 0; i < GRID_W * GRID_H; ++i) if (!isfinite(heightField[i])) heightField[i] = 0.0f;

    Camera3D cam = {0};
    cam.position = (Vector3){128.0f, 180.0f, 330.0f};
    cam.target   = (Vector3){128.0f, 0.0f, 128.0f};
    cam.up       = (Vector3){0.0f,1.0f,0.0f};
    cam.fovy     = 45.0f;
    cam.projection = CAMERA_PERSPECTIVE;

    float rotation = 0.0f;
    float distance = 320.0f;
    bool wireframe = false;

    Weather_Reset();

    float simTime = 0.0f; // simulated seconds

    while (!WindowShouldClose()) {
        float realDt = GetFrameTime();
        float dtSim  = realDt * TIME_SCALE;
        simTime += dtSim;

        // Input
        if (IsKeyPressed(KEY_W)) wireframe = !wireframe;
        if (IsKeyPressed(KEY_ONE))   { showTemp=1; showHum=0; showCloud=0; }
        if (IsKeyPressed(KEY_TWO))   { showTemp=0; showHum=1; showCloud=0; }
        if (IsKeyPressed(KEY_THREE)) { showTemp=0; showHum=0; showCloud=1; }
        if (IsKeyPressed(KEY_FOUR))  showWind = !showWind;
        if (IsKeyPressed(KEY_R))     { Weather_Reset(); simTime = 0.0f; }
        if (IsKeyDown(KEY_KP_ADD) || IsKeyDown(KEY_EQUAL)) TIME_SCALE *= 1.02f;
        if (IsKeyDown(KEY_KP_SUBTRACT) || IsKeyDown(KEY_MINUS)) TIME_SCALE /= 1.02f;
        if (TIME_SCALE < 1.0f) TIME_SCALE = 1.0f;
        if (TIME_SCALE > 5000.0f) TIME_SCALE = 5000.0f;

        if (IsKeyDown(KEY_LEFT))  rotation += 0.5f * realDt;
        if (IsKeyDown(KEY_RIGHT)) rotation -= 0.5f * realDt;
        distance -= GetMouseWheelMove() * 10.0f; distance = Clamp(distance, 80.0f, 1000.0f);
        float camX = sinf(rotation) * distance + cam.target.x;
        float camZ = cosf(rotation) * distance + cam.target.z;
        float camY = distance * 0.5f;
        cam.position = (Vector3){camX, camY, camZ};

        if (IsKeyDown(KEY_SPACE)) {
            int gx,gz; if (MouseToGrid(&gx,&gz, cam)) Weather_InjectMoisture(gx,gz, 10.0f);
        }

        Weather_Step(dtSim, simTime);

        BeginDrawing();
        ClearBackground((Color){10,10,14,255});

        BeginMode3D(cam);
            if (wireframe) {
                Mesh *m = &terrain.meshes[0];
                rlDisableTexture();
                rlBegin(RL_LINES);
                if (m->indices) {
                    for (int t = 0; t < m->triangleCount; ++t) {
                        int i0 = m->indices[t*3 + 0]; int i1 = m->indices[t*3 + 1]; int i2 = m->indices[t*3 + 2];
                        if (i0>=m->vertexCount||i1>=m->vertexCount||i2>=m->vertexCount) continue;
                        Vector3 v0 = {m->vertices[i0*3+0], m->vertices[i0*3+1], m->vertices[i0*3+2]};
                        Vector3 v1 = {m->vertices[i1*3+0], m->vertices[i1*3+1], m->vertices[i1*3+2]};
                        Vector3 v2 = {m->vertices[i2*3+0], m->vertices[i2*3+1], m->vertices[i2*3+2]};
                        rlColor4ub(140,140,140,255);
                        rlVertex3f(v0.x,v0.y,v0.z); rlVertex3f(v1.x,v1.y,v1.z);
                        rlVertex3f(v1.x,v1.y,v1.z); rlVertex3f(v2.x,v2.y,v2.z);
                        rlVertex3f(v2.x,v2.y,v2.z); rlVertex3f(v0.x,v0.y,v0.z);
                    }
                } else { // non-indexed fallback
                    int triVertCount = m->vertexCount - (m->vertexCount % 3);
                    for (int i=0;i<triVertCount;i+=3){
                        Vector3 v0={m->vertices[i*3+0],m->vertices[i*3+1],m->vertices[i*3+2]};
                        Vector3 v1={m->vertices[(i+1)*3+0],m->vertices[(i+1)*3+1],m->vertices[(i+1)*3+2]};
                        Vector3 v2={m->vertices[(i+2)*3+0],m->vertices[(i+2)*3+1],m->vertices[(i+2)*3+2]};
                        rlColor4ub(140,140,140,255);
                        rlVertex3f(v0.x,v0.y,v0.z); rlVertex3f(v1.x,v1.y,v1.z);
                        rlVertex3f(v1.x,v1.y,v1.z); rlVertex3f(v2.x,v2.y,v2.z);
                        rlVertex3f(v2.x,v2.y,v2.z); rlVertex3f(v0.x,v0.y,v0.z);
                    }
                }
                rlEnd();
            } else {
                DrawModel(terrain, (Vector3){0,0,0}, 1.0f, WHITE);
            }
            if (showTemp || showHum || showCloud) Weather_DrawOverlay();
            if (showWind) Weather_DrawWindVectors();
            Weather_DrawClouds();
        EndMode3D();

        DrawFPS(10,10);
        DrawText(TextFormat("TimeScale: %.1f x", TIME_SCALE), 10, 30, 18, LIGHTGRAY);
        DrawText(TextFormat("SimDayFrac: %.2f", fmodf(simTime/86400.0f,1.0f)), 10, 50, 18, LIGHTGRAY);
        DrawText("1:T 2:H 3:Cloud 4:Wind W:Wire SPACE:Moisture +/-:Time R:Reset", 10, 70, 18, SKYBLUE);
        const char *overlay = showTemp?"Temp":(showHum?"Humidity":(showCloud?"Cloud":"None"));
        DrawText(TextFormat("Overlay: %s", overlay), 10, 90, 18, GREEN);

        EndDrawing();
    }
    UnloadModel(terrain);
    CloseWindow();
    return 0;
}
