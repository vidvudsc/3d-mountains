#include "raylib.h"
#define RAYGUI_IMPLEMENTATION
#include "raygui.h"
#include "raymath.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#define MAP_SIZE 256
#define OCTAVES 4

#define EROSION_DROPLETS 1000000
#define EROSION_LIFETIME 2
#define EROSION_INERTIA 0.02f
#define EROSION_CAPACITY 3.0f
#define EROSION_MIN_SLOPE 0.001f    
#define EROSION_ERODE 0.04f 
#define EROSION_DEPOSIT 0.04f  
#define EROSION_EVAPORATE 0.1f
#define EROSION_MAX_STEP 1.0f     // limit per‐step removal

// --- Terrain control variables ---
float guiFreq = 0.004f;
float guiAmp = 1.0f;
float guiErodeRate = EROSION_ERODE;
float guiDepositRate = EROSION_DEPOSIT;
float guiInertia = EROSION_INERTIA;
int   guiLifetime = EROSION_LIFETIME;
float guiCapacity = EROSION_CAPACITY;
float guiMinSlope = EROSION_MIN_SLOPE;
float guiEvaporate = EROSION_EVAPORATE;
float guiMaxStep = EROSION_MAX_STEP;
int   guiDroplets = EROSION_DROPLETS;
float guiRockThreshold = 0.3f;  // Rock vs grass threshold

// Text buffers for value boxes
char guiFreqText[32] = "0.004";
char guiAmpText[32] = "1.0";
char guiErodeRateText[32] = "0.020";
char guiDepositRateText[32] = "0.040";
char guiInertiaText[32] = "0.02";
char guiLifetimeText[32] = "2";
char guiCapacityText[32] = "3.0";
char guiMinSlopeText[32] = "0.001";
char guiEvaporateText[32] = "0.1";
char guiMaxStepText[32] = "1.0";
char guiDropletsText[32] = "1000000";
char guiRockThresholdText[32] = "0.3";

// Edit mode flags
bool guiFreqEdit = false;
bool guiAmpEdit = false;
bool guiErodeRateEdit = false;
bool guiDepositRateEdit = false;
bool guiInertiaEdit = false;
bool guiLifetimeEdit = false;
bool guiCapacityEdit = false;
bool guiMinSlopeEdit = false;
bool guiEvaporateEdit = false;
bool guiMaxStepEdit = false;
bool guiDropletsEdit = false;
bool guiRockThresholdEdit = false;

// --- HYDRO EROSION PARAMETERS ---
#define EROSION_DROPLETS 700000
#define EROSION_CAPACITY 2.0f
#define EROSION_MIN_SLOPE 0.001f    
#define EROSION_EVAPORATE 0.1f
#define EROSION_MAX_STEP 1.0f     // limit per‐step removal

#define MAX_DROPLET_PATH 1200
static Vector3 dropletPath[MAX_DROPLET_PATH];
static int dropletPathLen = 0;

static float heightmap[MAP_SIZE][MAP_SIZE];
static float heightmap_perlin[MAP_SIZE][MAP_SIZE];
static float heightmap_eroded[MAP_SIZE][MAP_SIZE];
static int   heatCount[MAP_SIZE][MAP_SIZE]; // droplet visit counts for heatmap

static float Smoothstep(float t) {
    return t * t * (3.0f - 2.0f * t);
}

static int Hash(int x, int y) {
    int h = x * 3747261393 + y * 668265263;
    h = (h ^ (h >> 13)) * 1274126177;
    return h & 1023;
}
static float Gradient(int hash, float x, float y) {
    switch (hash & 3) {
        case 0: return  x + y;
        case 1: return -x + y;
        case 2: return  x - y;
        default: return -x - y;
    }
}

static float Perlin(float x, float y) {
    int X = (int)floorf(x) & 255;
    int Y = (int)floorf(y) & 255;
    float xf = x - floorf(x);
    float yf = y - floorf(y);

    int h00 = Hash(X, Y);
    int h10 = Hash(X + 1, Y);
    int h01 = Hash(X, Y + 1);
    int h11 = Hash(X + 1, Y + 1);

    float u = Smoothstep(xf);
    float v = Smoothstep(yf);

    float n00 = Gradient(h00, xf, yf);
    float n10 = Gradient(h10, xf - 1, yf);
    float n01 = Gradient(h01, xf, yf - 1);
    float n11 = Gradient(h11, xf - 1, yf - 1);

    return Lerp(Lerp(n00, n10, u), Lerp(n01, n11, u), v);
}

static void GenerateLayeredNoiseHeightmap(float map[MAP_SIZE][MAP_SIZE]) {
    const float offsetX = (float)GetRandomValue(0, 10000);
    const float offsetY = (float)GetRandomValue(0, 10000);

    for (int z = 0; z < MAP_SIZE; z++)
        for (int x = 0; x < MAP_SIZE; x++) {
            float noise = 0.0f;
            float frequency = guiFreq;
            float amplitude = guiAmp;
            frequency *= 1.9f;
            float totalAmp = 0.0f;

            for (int o = 0; o < OCTAVES; o++) {
                float nx = (x + offsetX) * frequency;
                float ny = (z + offsetY) * frequency;
                noise += Perlin(nx, ny) * amplitude;
                totalAmp += amplitude;
                amplitude *= 0.5f;
                frequency *= 1.9f;
            }

            noise /= totalAmp;
            map[x][z] = (noise + 1.0f) * 0.5f;
        }

    float minH = 1e9f, maxH = -1e9f;
    for (int z = 0; z < MAP_SIZE; z++)
        for (int x = 0; x < MAP_SIZE; x++) {
            map[x][z] = powf(map[x][z], 1.5f);
            if (map[x][z] < minH) minH = map[x][z];
            if (map[x][z] > maxH) maxH = map[x][z];
        }

    for (int z = 0; z < MAP_SIZE; z++)
        for (int x = 0; x < MAP_SIZE; x++)
            map[x][z] = (map[x][z] - minH) / (maxH - minH);
}

static void ApplyHydroErosion(float map[MAP_SIZE][MAP_SIZE]) {
    // reset heatmap counters
    for (int z = 0; z < MAP_SIZE; z++)
        for (int x = 0; x < MAP_SIZE; x++)
            heatCount[x][z] = 0;
    for (int n = 0; n < guiDroplets; n++) {
        float x = (float)(rand() % (MAP_SIZE - 1)) + 0.5f;
        float y = (float)(rand() % (MAP_SIZE - 1)) + 0.5f;
        float dirX = 0, dirY = 0;
        float speed = 1.0f;
        float water = 1.0f;
        float sediment = 0.0f;

        for (int lifetime = 0; lifetime < guiLifetime; lifetime++) {
            int xi = (int)x;
            int yi = (int)y;
            heatCount[xi][yi]++; // accumulate visits for heatmap
            if (xi < 0 || xi >= MAP_SIZE - 1 || yi < 0 || yi >= MAP_SIZE - 1) break;

            // Calculate height and gradient using bilinear interpolation
            float fx = x - xi;
            float fy = y - yi;
            float h00 = map[xi][yi];
            float h10 = map[xi+1][yi];
            float h01 = map[xi][yi+1];
            float h11 = map[xi+1][yi+1];
            float height = h00 * (1-fx)*(1-fy) + h10 * fx*(1-fy) + h01 * (1-fx)*fy + h11 * fx*fy;
            float gradX = (h10 - h00) * (1-fy) + (h11 - h01) * fy;
            float gradY = (h01 - h00) * (1-fx) + (h11 - h10) * fx;

            // Update direction with inertia
            dirX = dirX * guiInertia - gradX * (1-guiInertia);
            dirY = dirY * guiInertia - gradY * (1-guiInertia);
            float len = sqrtf(dirX*dirX + dirY*dirY);
            if (len != 0) { dirX /= len; dirY /= len; }

            x += dirX;
            y += dirY;

            // Stop if out of bounds
            if (x < 0 || x >= MAP_SIZE - 1 || y < 0 || y >= MAP_SIZE - 1) break;

            // Compute new height
            xi = (int)x;
            yi = (int)y;
            fx = x - xi;
            fy = y - yi;
            float nh00 = map[xi][yi];
            float nh10 = map[xi+1][yi];
            float nh01 = map[xi][yi+1];
            float nh11 = map[xi+1][yi+1];
            float newHeight = nh00 * (1-fx)*(1-fy) + nh10 * fx*(1-fy) + nh01 * (1-fx)*fy + nh11 * fx*fy;
            float deltaH = newHeight - height;

            // Calculate sediment capacity
            float capacity = fmaxf(-deltaH * speed * water * guiCapacity, guiMinSlope);

            // Erode or deposit
            if (sediment > capacity || deltaH > 0) {
                float deposit = (deltaH > 0) ? fminf(sediment, deltaH) : (sediment - capacity) * guiDepositRate;
                deposit = fminf(deposit, guiMaxStep); // cap deposit
                sediment -= deposit;
                // Deposit sediment
                map[xi][yi] = fminf(fmaxf(map[xi][yi] + deposit * (1-fx)*(1-fy), 0.0f), 1.0f);
                map[xi+1][yi] = fminf(fmaxf(map[xi+1][yi] + deposit * fx*(1-fy), 0.0f), 1.0f);
                map[xi][yi+1] = fminf(fmaxf(map[xi][yi+1] + deposit * (1-fx)*fy, 0.0f), 1.0f);
                map[xi+1][yi+1] = fminf(fmaxf(map[xi+1][yi+1] + deposit * fx*fy, 0.0f), 1.0f);
            } else {
                float erode = fminf((capacity - sediment) * guiErodeRate, -deltaH);
                erode = fminf(erode, guiMaxStep); // cap erosion
                for (int dx = 0; dx <= 1; dx++) for (int dy = 0; dy <= 1; dy++) {
                    float w = ((dx == 0) ? (1-fx) : fx) * ((dy == 0) ? (1-fy) : fy);
                    float d = fminf(map[xi+dx][yi+dy], erode * w);
                    map[xi+dx][yi+dy] = fminf(fmaxf(map[xi+dx][yi+dy] - d, 0.0f), 1.0f);
                    sediment += d;
                }
            }

            speed = sqrtf(speed*speed + deltaH * 0.1f);
            water *= (1.0f - guiEvaporate);
            if (water < 0.01f) break;
        }
    }
}

// Simple box blur smoothing
static void SmoothHeightmap(float map[MAP_SIZE][MAP_SIZE], int passes) {
    float temp[MAP_SIZE][MAP_SIZE];
    for (int p = 0; p < passes; p++) {
        for (int z = 1; z < MAP_SIZE - 1; z++) {
            for (int x = 1; x < MAP_SIZE - 1; x++) {
                float sum = 0.0f;
                for (int dz = -1; dz <= 1; dz++)
                    for (int dx = -1; dx <= 1; dx++)
                        sum += map[x + dx][z + dz];
                temp[x][z] = sum / 9.0f;
            }
        }
        // Copy back
        for (int z = 1; z < MAP_SIZE - 1; z++)
            for (int x = 1; x < MAP_SIZE - 1; x++)
                map[x][z] = temp[x][z];
    }
}

static Color color(Vector3 n) {
    const Vector3 lightDir = { 0.4f, 1.0f, 0.5f };
    float shade = Clamp(Vector3DotProduct(Vector3Normalize(n), Vector3Normalize(lightDir)), 0.0f, 1.0f);
    
    // Calculate slope angle (how steep the surface is)
    float slope = 1.0f - Vector3DotProduct(Vector3Normalize(n), (Vector3){ 0.0f, 1.0f, 0.0f });
    slope = Clamp(slope, 0.0f, 1.0f);
    
    // Define terrain types based on slope
    // Use the GUI threshold value instead of constant
    Color terrainColor;
    if (slope > guiRockThreshold) {
        // Rock - gray/brown colors
        unsigned char baseR = 80 + (unsigned char)(slope * 40);
        unsigned char baseG = 70 + (unsigned char)(slope * 30);
        unsigned char baseB = 60 + (unsigned char)(slope * 20);
        terrainColor = (Color){ baseR, baseG, baseB, 255 };
    } else {
        // Grass - green colors
        unsigned char baseR = 40 + (unsigned char)(slope * 20);
        unsigned char baseG = 80 + (unsigned char)(slope * 60);
        unsigned char baseB = 30 + (unsigned char)(slope * 15);
        terrainColor = (Color){ baseR, baseG, baseB, 255 };
    }
    
    // Apply lighting/shading
    unsigned char r = (unsigned char)(terrainColor.r * (0.3f + 0.7f * shade));
    unsigned char g = (unsigned char)(terrainColor.g * (0.3f + 0.7f * shade));
    unsigned char b = (unsigned char)(terrainColor.b * (0.3f + 0.7f * shade));
    
    return (Color){ r, g, b, 255 };
}

static Model BuildTerrainModel(float map[MAP_SIZE][MAP_SIZE]) {
    const int vertsCount = MAP_SIZE * MAP_SIZE;
    const int indicesCount = (MAP_SIZE - 1) * (MAP_SIZE - 1) * 6;

    Vector3 *verts = MemAlloc(sizeof(Vector3) * vertsCount);
    Vector3 *normals = MemAlloc(sizeof(Vector3) * vertsCount);
    Color   *colBuf = MemAlloc(sizeof(Color) * vertsCount);
    int     *indices = MemAlloc(sizeof(int) * indicesCount);

    for (int z = 0; z < MAP_SIZE; z++)
        for (int x = 0; x < MAP_SIZE; x++) {
            int i = z * MAP_SIZE + x;
            float h = map[x][z];
            verts[i] = (Vector3){ (float)x, h * 120.0f, (float)z };
        }

    int idx = 0;
    for (int z = 0; z < MAP_SIZE - 1; z++)
        for (int x = 0; x < MAP_SIZE - 1; x++) {
            int i0 = z * MAP_SIZE + x;
            int i1 = i0 + 1;
            int i2 = i0 + MAP_SIZE;
            int i3 = i2 + 1;
            indices[idx++] = i0; indices[idx++] = i2; indices[idx++] = i1;
            indices[idx++] = i1; indices[idx++] = i2; indices[idx++] = i3;
        }

    for (int i = 0; i < vertsCount; i++) normals[i] = (Vector3){0};

    for (int i = 0; i < indicesCount; i += 3) {
        int i0 = indices[i];
        int i1 = indices[i + 1];
        int i2 = indices[i + 2];
        Vector3 v0 = verts[i0];
        Vector3 v1 = verts[i1];
        Vector3 v2 = verts[i2];
        Vector3 n = Vector3Normalize(Vector3CrossProduct(Vector3Subtract(v1, v0), Vector3Subtract(v2, v0)));
        normals[i0] = Vector3Add(normals[i0], n);
        normals[i1] = Vector3Add(normals[i1], n);
        normals[i2] = Vector3Add(normals[i2], n);
    }

    for (int i = 0; i < vertsCount; i++) {
        normals[i] = Vector3Normalize(normals[i]);
        colBuf[i] = color(normals[i]);
    }

    Mesh mesh = { 0 };
    mesh.vertexCount = vertsCount;
    mesh.triangleCount = indicesCount / 3;
    mesh.vertices = (float *)MemAlloc(vertsCount * 3 * sizeof(float));
    mesh.normals = (float *)MemAlloc(vertsCount * 3 * sizeof(float));
    mesh.colors = (unsigned char *)MemAlloc(vertsCount * 4 * sizeof(unsigned char));
    mesh.indices = (unsigned short *)MemAlloc(indicesCount * sizeof(unsigned short));

    for (int v = 0; v < vertsCount; v++) {
        mesh.vertices[v * 3 + 0] = verts[v].x;
        mesh.vertices[v * 3 + 1] = verts[v].y;
        mesh.vertices[v * 3 + 2] = verts[v].z;

        mesh.normals[v * 3 + 0] = normals[v].x;
        mesh.normals[v * 3 + 1] = normals[v].y;
        mesh.normals[v * 3 + 2] = normals[v].z;

        mesh.colors[v * 4 + 0] = colBuf[v].r;
        mesh.colors[v * 4 + 1] = colBuf[v].g;
        mesh.colors[v * 4 + 2] = colBuf[v].b;
        mesh.colors[v * 4 + 3] = colBuf[v].a;
    }

    for (int k = 0; k < indicesCount; k++) mesh.indices[k] = indices[k];

    UploadMesh(&mesh, true);

    MemFree(verts);
    MemFree(normals);
    MemFree(colBuf);
    MemFree(indices);

    return LoadModelFromMesh(mesh);
}

// Simulate a single droplet path (does not modify heightmap)
static void SimulateDropletPath(const float map[MAP_SIZE][MAP_SIZE], Vector3 *outPath, int *outLen) {
    float x = (float)(rand() % (MAP_SIZE - 1)) + 0.5f;
    float y = (float)(rand() % (MAP_SIZE - 1)) + 0.5f;
    float dirX = 0, dirY = 0;
    float speed = 1.0f;
    float water = 1.0f;
    int len = 0;
    for (int lifetime = 0; lifetime < MAX_DROPLET_PATH; lifetime++) {
        int xi = (int)x;
        int yi = (int)y;
        if (xi < 0 || xi >= MAP_SIZE - 1 || yi < 0 || yi >= MAP_SIZE - 1) break;
        float fx = x - xi;
        float fy = y - yi;
        float h00 = map[xi][yi];
        float h10 = map[xi+1][yi];
        float h01 = map[xi][yi+1];
        float h11 = map[xi+1][yi+1];
        float height = h00 * (1-fx)*(1-fy) + h10 * fx*(1-fy) + h01 * (1-fx)*fy + h11 * fx*fy;
        if (len < MAX_DROPLET_PATH) {
            outPath[len++] = (Vector3){ x, height * 80.0f, y };
        }
        float gradX = (h10 - h00) * (1-fy) + (h11 - h01) * fy;
        float gradY = (h01 - h00) * (1-fx) + (h11 - h10) * fx;
        // No inertia for visualization: always steepest descent
        float lenDir = sqrtf(gradX*gradX + gradY*gradY);
        if (lenDir != 0) { gradX /= lenDir; gradY /= lenDir; }
        dirX = -gradX;
        dirY = -gradY;
        x += dirX;
        y += dirY;
        if (x < 0 || x >= MAP_SIZE - 1 || y < 0 || y >= MAP_SIZE - 1) break;
        speed = sqrtf(speed*speed + (height - outPath[len-1].y/80.0f) * 0.1f);
        water *= (1.0f - guiEvaporate);
        if (water < 0.01f) break;
    }
    *outLen = len;
}

// Write mesh as OBJ file
static void ExportMeshOBJ(const float map[MAP_SIZE][MAP_SIZE], const char *filename) {
    // Compute normals as in BuildTerrainModel
    const int vertsCount = MAP_SIZE * MAP_SIZE;
    const int indicesCount = (MAP_SIZE - 1) * (MAP_SIZE - 1) * 6;
    Vector3 *verts = MemAlloc(sizeof(Vector3) * vertsCount);
    Vector3 *normals = MemAlloc(sizeof(Vector3) * vertsCount);
    Color   *colBuf = MemAlloc(sizeof(Color) * vertsCount);
    int     *indices = MemAlloc(sizeof(int) * indicesCount);
    for (int z = 0; z < MAP_SIZE; z++)
        for (int x = 0; x < MAP_SIZE; x++) {
            int i = z * MAP_SIZE + x;
            float h = map[x][z];
            verts[i] = (Vector3){ (float)x, h * 80.0f, (float)z };
        }
    int idx = 0;
    for (int z = 0; z < MAP_SIZE - 1; z++)
        for (int x = 0; x < MAP_SIZE - 1; x++) {
            int i0 = z * MAP_SIZE + x;
            int i1 = i0 + 1;
            int i2 = i0 + MAP_SIZE;
            int i3 = i2 + 1;
            indices[idx++] = i0; indices[idx++] = i2; indices[idx++] = i1;
            indices[idx++] = i1; indices[idx++] = i2; indices[idx++] = i3;
        }
    for (int i = 0; i < vertsCount; i++) normals[i] = (Vector3){0};
    for (int i = 0; i < indicesCount; i += 3) {
        int i0 = indices[i];
        int i1 = indices[i + 1];
        int i2 = indices[i + 2];
        Vector3 v0 = verts[i0];
        Vector3 v1 = verts[i1];
        Vector3 v2 = verts[i2];
        Vector3 n = Vector3Normalize(Vector3CrossProduct(Vector3Subtract(v1, v0), Vector3Subtract(v2, v0)));
        normals[i0] = Vector3Add(normals[i0], n);
        normals[i1] = Vector3Add(normals[i1], n);
        normals[i2] = Vector3Add(normals[i2], n);
    }
    for (int i = 0; i < vertsCount; i++) normals[i] = Vector3Normalize(normals[i]);
    for (int i = 0; i < vertsCount; i++) colBuf[i] = color(normals[i]);
    FILE *f = fopen(filename, "w");
    if (!f) { MemFree(verts); MemFree(normals); MemFree(colBuf); MemFree(indices); return; }
    // Write vertices
    for (int i = 0; i < vertsCount; i++) {
        fprintf(f, "v %f %f %f\n", verts[i].x, verts[i].y, verts[i].z);
    }
    // Write normals
    for (int i = 0; i < vertsCount; i++) {
        fprintf(f, "vn %f %f %f\n", normals[i].x, normals[i].y, normals[i].z);
    }
    // Optionally, write color as comment per vertex
    for (int i = 0; i < vertsCount; i++) {
        fprintf(f, "# color %d %d %d\n", colBuf[i].r, colBuf[i].g, colBuf[i].b);
    }
    // Write faces (OBJ is 1-based)
    for (int i = 0; i < indicesCount; i += 3) {
        int i0 = indices[i] + 1;
        int i1 = indices[i+1] + 1;
        int i2 = indices[i+2] + 1;
        // OBJ face with normal indices: f v//vn v//vn v//vn
        fprintf(f, "f %d//%d %d//%d %d//%d\n", i0, i0, i1, i1, i2, i2);
    }
    fclose(f);
    MemFree(verts);
    MemFree(normals);
    MemFree(colBuf);
    MemFree(indices);
}

int main(void) {
    InitWindow(1200, 800, "Terrain Generator");
    SetTargetFPS(60);

    srand((unsigned int)time(NULL));

    GenerateLayeredNoiseHeightmap(heightmap_perlin);
    // Copy perlin to eroded
    for (int z = 0; z < MAP_SIZE; z++)
        for (int x = 0; x < MAP_SIZE; x++)
            heightmap_eroded[x][z] = heightmap_perlin[x][z];
    ApplyHydroErosion(heightmap_eroded);
    // Extra smoothing
    SmoothHeightmap(heightmap_eroded, 8);

    // Multiple pit-filling passes with epsilon
    #define PIT_EPSILON 1e-5f
    for (int pass = 0; pass < 3; pass++) {
        for (int z = 1; z < MAP_SIZE-1; z++) {
            for (int x = 1; x < MAP_SIZE-1; x++) {
                float h = heightmap_eroded[x][z];
                float m = h;
                for (int dz = -1; dz <= 1; dz++)
                    for (int dx = -1; dx <= 1; dx++)
                        if (heightmap_eroded[x+dx][z+dz] < m)
                            m = heightmap_eroded[x+dx][z+dz];
                if (h < m - PIT_EPSILON) heightmap_eroded[x][z] = m;
            }
        }
    }

    // Final clamp to [0, 1]
    for (int z = 0; z < MAP_SIZE; z++)
        for (int x = 0; x < MAP_SIZE; x++) {
            if (heightmap_eroded[x][z] < 0.0f) heightmap_eroded[x][z] = 0.0f;
            else if (heightmap_eroded[x][z] > 1.0f) heightmap_eroded[x][z] = 1.0f;
        }
    Model terrain_perlin = BuildTerrainModel(heightmap_perlin);
    Model terrain_eroded = BuildTerrainModel(heightmap_eroded);
    int showEroded = 0; // 0: show perlin, 1: show eroded

    Camera3D cam = {
        .position = { MAP_SIZE / 2.0f, 120.0f, MAP_SIZE * 1.3f },
        .target = { MAP_SIZE / 2.0f, 0.0f, MAP_SIZE / 2.0f },
        .up = { 0.0f, 1.0f, 0.0f },
        .fovy = 45.0f,
        .projection = CAMERA_PERSPECTIVE
    };

    float rotation = 0.0f;
    float distance = 300.0f;
    bool showPanel = false;  // Start with panel hidden
    Rectangle panelBounds = { 900, 40, 280, 400 };
    Rectangle panelButton = { 1100, 10, 80, 25 };  // Button in top right corner

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_R)) {
            GenerateLayeredNoiseHeightmap(heightmap_perlin);
            for (int z = 0; z < MAP_SIZE; z++)
                for (int x = 0; x < MAP_SIZE; x++)
                    heightmap_eroded[x][z] = heightmap_perlin[x][z];
            ApplyHydroErosion(heightmap_eroded);
            SmoothHeightmap(heightmap_eroded, 8);
            UnloadModel(terrain_perlin);
            UnloadModel(terrain_eroded);
            terrain_perlin = BuildTerrainModel(heightmap_perlin);
            terrain_eroded = BuildTerrainModel(heightmap_eroded);
            showEroded = 0;
        }

        if (IsKeyPressed(KEY_MINUS)) showEroded = 0;
        if (IsKeyPressed(KEY_EQUAL)) {
            showEroded = 1;
            // Regenerate eroded terrain with current GUI values
            for (int z = 0; z < MAP_SIZE; z++)
                for (int x = 0; x < MAP_SIZE; x++)
                    heightmap_eroded[x][z] = heightmap_perlin[x][z];
            ApplyHydroErosion(heightmap_eroded);
            SmoothHeightmap(heightmap_eroded, 8);
            UnloadModel(terrain_eroded);
            terrain_eroded = BuildTerrainModel(heightmap_eroded);
        }
        if (IsKeyPressed(KEY_D)) {
            // Show droplet path for current state
            const float (*map)[MAP_SIZE] = showEroded ? heightmap_eroded : heightmap_perlin;
            SimulateDropletPath(map, dropletPath, &dropletPathLen);
        }
        if (IsKeyPressed(KEY_S) && showEroded) {
            ExportMeshOBJ(heightmap_eroded, "eroded_terrain.obj");
        }

        if (IsKeyDown(KEY_LEFT)) rotation += 0.01f;
        if (IsKeyDown(KEY_RIGHT)) rotation -= 0.01f;
        distance -= GetMouseWheelMove() * 5.0f;
        distance = Clamp(distance, 50.0f, 800.0f);

        float camX = sinf(rotation) * distance;
        float camZ = cosf(rotation) * distance;
        float camY = distance * 0.5f;
        cam.position = Vector3Add(cam.target, (Vector3){ camX, camY, camZ });

        BeginDrawing();
            ClearBackground(BLACK);
            BeginMode3D(cam);
                DrawModel(showEroded ? terrain_eroded : terrain_perlin, (Vector3){ 0 }, 1.0f, WHITE);
                // Draw droplet path if present
                if (dropletPathLen > 1) {
                    for (int i = 0; i < dropletPathLen - 1; i++) {
                        DrawLine3D(dropletPath[i], dropletPath[i+1], RED);
                        DrawSphere(dropletPath[i], 0.7f, YELLOW);
                    }
                    DrawSphere(dropletPath[dropletPathLen-1], 1.2f, ORANGE); // end point
                }
            EndMode3D();

            // Draw panel button in top right corner (only when panel is hidden)
            if (!showPanel) {
                if (GuiButton(panelButton, "Panel")) {
                    showPanel = !showPanel;
                }
            }

            if (showPanel) {
                // Add close button (X) in top right of panel
                Rectangle closeButton = { panelBounds.x + panelBounds.width - 25, panelBounds.y + 5, 20, 20 };
                if (GuiButton(closeButton, "X")) {
                    showPanel = false;
                }
                
                GuiWindowBox(panelBounds, "Terrain Settings");
                int y = panelBounds.y + 30;

                // Update text buffers from current values
                if (!guiFreqEdit) sprintf(guiFreqText, "%.4f", guiFreq);
                if (!guiAmpEdit) sprintf(guiAmpText, "%.2f", guiAmp);
                if (!guiInertiaEdit) sprintf(guiInertiaText, "%.4f", guiInertia);
                if (!guiErodeRateEdit) sprintf(guiErodeRateText, "%.3f", guiErodeRate);
                if (!guiDepositRateEdit) sprintf(guiDepositRateText, "%.3f", guiDepositRate);
                if (!guiCapacityEdit) sprintf(guiCapacityText, "%.1f", guiCapacity);
                if (!guiMinSlopeEdit) sprintf(guiMinSlopeText, "%.3f", guiMinSlope);
                if (!guiEvaporateEdit) sprintf(guiEvaporateText, "%.1f", guiEvaporate);
                if (!guiMaxStepEdit) sprintf(guiMaxStepText, "%.1f", guiMaxStep);
                if (!guiRockThresholdEdit) sprintf(guiRockThresholdText, "%.2f", guiRockThreshold);

                GuiLabel((Rectangle){ panelBounds.x + 10, y, 100, 20 }, "Perlin Freq:");
                if (GuiValueBoxFloat((Rectangle){ panelBounds.x + 120, y, 100, 20 }, NULL, guiFreqText, &guiFreq, guiFreqEdit)) {
                    guiFreqEdit = !guiFreqEdit;
                    // Auto-regenerate when frequency changes
                    if (!guiFreqEdit) {
                        GenerateLayeredNoiseHeightmap(heightmap_perlin);
                        for (int z = 0; z < MAP_SIZE; z++)
                            for (int x = 0; x < MAP_SIZE; x++)
                                heightmap_eroded[x][z] = heightmap_perlin[x][z];
                        ApplyHydroErosion(heightmap_eroded);
                        SmoothHeightmap(heightmap_eroded, 8);
                        UnloadModel(terrain_perlin);
                        UnloadModel(terrain_eroded);
                        terrain_perlin = BuildTerrainModel(heightmap_perlin);
                        terrain_eroded = BuildTerrainModel(heightmap_eroded);
                    }
                }
                y += 30;

                GuiLabel((Rectangle){ panelBounds.x + 10, y, 100, 20 }, "Amplitude:");
                if (GuiValueBoxFloat((Rectangle){ panelBounds.x + 120, y, 100, 20 }, NULL, guiAmpText, &guiAmp, guiAmpEdit)) {
                    guiAmpEdit = !guiAmpEdit;
                    // Auto-regenerate when amplitude changes
                    if (!guiAmpEdit) {
                        GenerateLayeredNoiseHeightmap(heightmap_perlin);
                        for (int z = 0; z < MAP_SIZE; z++)
                            for (int x = 0; x < MAP_SIZE; x++)
                                heightmap_eroded[x][z] = heightmap_perlin[x][z];
                        ApplyHydroErosion(heightmap_eroded);
                        SmoothHeightmap(heightmap_eroded, 8);
                        UnloadModel(terrain_perlin);
                        UnloadModel(terrain_eroded);
                        terrain_perlin = BuildTerrainModel(heightmap_perlin);
                        terrain_eroded = BuildTerrainModel(heightmap_eroded);
                    }
                }
                y += 30;

                GuiLabel((Rectangle){ panelBounds.x + 10, y, 100, 20 }, "Inertia:");
                if (GuiValueBoxFloat((Rectangle){ panelBounds.x + 120, y, 100, 20 }, NULL, guiInertiaText, &guiInertia, guiInertiaEdit)) {
                    guiInertiaEdit = !guiInertiaEdit;
                    // Auto-regenerate erosion when inertia changes
                    if (!guiInertiaEdit) {
                        for (int z = 0; z < MAP_SIZE; z++)
                            for (int x = 0; x < MAP_SIZE; x++)
                                heightmap_eroded[x][z] = heightmap_perlin[x][z];
                        ApplyHydroErosion(heightmap_eroded);
                        SmoothHeightmap(heightmap_eroded, 8);
                        UnloadModel(terrain_eroded);
                        terrain_eroded = BuildTerrainModel(heightmap_eroded);
                    }
                }
                y += 30;

                GuiLabel((Rectangle){ panelBounds.x + 10, y, 100, 20 }, "Erode Rate:");
                if (GuiValueBoxFloat((Rectangle){ panelBounds.x + 120, y, 100, 20 }, NULL, guiErodeRateText, &guiErodeRate, guiErodeRateEdit)) {
                    guiErodeRateEdit = !guiErodeRateEdit;
                    // Auto-regenerate erosion when erode rate changes
                    if (!guiErodeRateEdit) {
                        for (int z = 0; z < MAP_SIZE; z++)
                            for (int x = 0; x < MAP_SIZE; x++)
                                heightmap_eroded[x][z] = heightmap_perlin[x][z];
                        ApplyHydroErosion(heightmap_eroded);
                        SmoothHeightmap(heightmap_eroded, 8);
                        UnloadModel(terrain_eroded);
                        terrain_eroded = BuildTerrainModel(heightmap_eroded);
                    }
                }
                y += 30;

                GuiLabel((Rectangle){ panelBounds.x + 10, y, 100, 20 }, "Deposit Rate:");
                if (GuiValueBoxFloat((Rectangle){ panelBounds.x + 120, y, 100, 20 }, NULL, guiDepositRateText, &guiDepositRate, guiDepositRateEdit)) {
                    guiDepositRateEdit = !guiDepositRateEdit;
                    // Auto-regenerate erosion when deposit rate changes
                    if (!guiDepositRateEdit) {
                        for (int z = 0; z < MAP_SIZE; z++)
                            for (int x = 0; x < MAP_SIZE; x++)
                                heightmap_eroded[x][z] = heightmap_perlin[x][z];
                        ApplyHydroErosion(heightmap_eroded);
                        SmoothHeightmap(heightmap_eroded, 8);
                        UnloadModel(terrain_eroded);
                        terrain_eroded = BuildTerrainModel(heightmap_eroded);
                    }
                }
                y += 30;

                GuiLabel((Rectangle){ panelBounds.x + 10, y, 100, 20 }, "Lifetime:");
                if (GuiValueBox((Rectangle){ panelBounds.x + 120, y, 100, 20 }, NULL, &guiLifetime, 1, 10, guiLifetimeEdit)) {
                    guiLifetimeEdit = !guiLifetimeEdit;
                    // Auto-regenerate erosion when lifetime changes
                    if (!guiLifetimeEdit) {
                        for (int z = 0; z < MAP_SIZE; z++)
                            for (int x = 0; x < MAP_SIZE; x++)
                                heightmap_eroded[x][z] = heightmap_perlin[x][z];
                        ApplyHydroErosion(heightmap_eroded);
                        SmoothHeightmap(heightmap_eroded, 8);
                        UnloadModel(terrain_eroded);
                        terrain_eroded = BuildTerrainModel(heightmap_eroded);
                    }
                }
                y += 30;

                GuiLabel((Rectangle){ panelBounds.x + 10, y, 100, 20 }, "Capacity:");
                if (GuiValueBoxFloat((Rectangle){ panelBounds.x + 120, y, 100, 20 }, NULL, guiCapacityText, &guiCapacity, guiCapacityEdit)) {
                    guiCapacityEdit = !guiCapacityEdit;
                    // Auto-regenerate erosion when capacity changes
                    if (!guiCapacityEdit) {
                        for (int z = 0; z < MAP_SIZE; z++)
                            for (int x = 0; x < MAP_SIZE; x++)
                                heightmap_eroded[x][z] = heightmap_perlin[x][z];
                        ApplyHydroErosion(heightmap_eroded);
                        SmoothHeightmap(heightmap_eroded, 8);
                        UnloadModel(terrain_eroded);
                        terrain_eroded = BuildTerrainModel(heightmap_eroded);
                    }
                }
                y += 30;

                GuiLabel((Rectangle){ panelBounds.x + 10, y, 100, 20 }, "Min Slope:");
                if (GuiValueBoxFloat((Rectangle){ panelBounds.x + 120, y, 100, 20 }, NULL, guiMinSlopeText, &guiMinSlope, guiMinSlopeEdit)) {
                    guiMinSlopeEdit = !guiMinSlopeEdit;
                    // Auto-regenerate erosion when min slope changes
                    if (!guiMinSlopeEdit) {
                        for (int z = 0; z < MAP_SIZE; z++)
                            for (int x = 0; x < MAP_SIZE; x++)
                                heightmap_eroded[x][z] = heightmap_perlin[x][z];
                        ApplyHydroErosion(heightmap_eroded);
                        SmoothHeightmap(heightmap_eroded, 8);
                        UnloadModel(terrain_eroded);
                        terrain_eroded = BuildTerrainModel(heightmap_eroded);
                    }
                }
                y += 30;

                GuiLabel((Rectangle){ panelBounds.x + 10, y, 100, 20 }, "Evaporate:");
                if (GuiValueBoxFloat((Rectangle){ panelBounds.x + 120, y, 100, 20 }, NULL, guiEvaporateText, &guiEvaporate, guiEvaporateEdit)) {
                    guiEvaporateEdit = !guiEvaporateEdit;
                    // Auto-regenerate erosion when evaporate changes
                    if (!guiEvaporateEdit) {
                        for (int z = 0; z < MAP_SIZE; z++)
                            for (int x = 0; x < MAP_SIZE; x++)
                                heightmap_eroded[x][z] = heightmap_perlin[x][z];
                        ApplyHydroErosion(heightmap_eroded);
                        SmoothHeightmap(heightmap_eroded, 8);
                        UnloadModel(terrain_eroded);
                        terrain_eroded = BuildTerrainModel(heightmap_eroded);
                    }
                }
                y += 30;

                GuiLabel((Rectangle){ panelBounds.x + 10, y, 100, 20 }, "Max Step:");
                if (GuiValueBoxFloat((Rectangle){ panelBounds.x + 120, y, 100, 20 }, NULL, guiMaxStepText, &guiMaxStep, guiMaxStepEdit)) {
                    guiMaxStepEdit = !guiMaxStepEdit;
                    // Auto-regenerate erosion when max step changes
                    if (!guiMaxStepEdit) {
                        for (int z = 0; z < MAP_SIZE; z++)
                            for (int x = 0; x < MAP_SIZE; x++)
                                heightmap_eroded[x][z] = heightmap_perlin[x][z];
                        ApplyHydroErosion(heightmap_eroded);
                        SmoothHeightmap(heightmap_eroded, 8);
                        UnloadModel(terrain_eroded);
                        terrain_eroded = BuildTerrainModel(heightmap_eroded);
                    }
                }
                y += 30;

                GuiLabel((Rectangle){ panelBounds.x + 10, y, 100, 20 }, "Droplets:");
                if (GuiValueBox((Rectangle){ panelBounds.x + 120, y, 100, 20 }, NULL, &guiDroplets, 1000, 2000000, guiDropletsEdit)) {
                    guiDropletsEdit = !guiDropletsEdit;
                    // Auto-regenerate erosion when droplets changes
                    if (!guiDropletsEdit) {
                        for (int z = 0; z < MAP_SIZE; z++)
                            for (int x = 0; x < MAP_SIZE; x++)
                                heightmap_eroded[x][z] = heightmap_perlin[x][z];
                        ApplyHydroErosion(heightmap_eroded);
                        SmoothHeightmap(heightmap_eroded, 8);
                        UnloadModel(terrain_eroded);
                        terrain_eroded = BuildTerrainModel(heightmap_eroded);
                    }
                }
                y += 30;

                GuiLabel((Rectangle){ panelBounds.x + 10, y, 100, 20 }, "Rock Threshold:");
                if (GuiValueBoxFloat((Rectangle){ panelBounds.x + 120, y, 100, 20 }, NULL, guiRockThresholdText, &guiRockThreshold, guiRockThresholdEdit)) {
                    guiRockThresholdEdit = !guiRockThresholdEdit;
                    // Auto-update terrain models when rock threshold changes (no regeneration needed)
                    if (!guiRockThresholdEdit) {
                        UnloadModel(terrain_perlin);
                        UnloadModel(terrain_eroded);
                        terrain_perlin = BuildTerrainModel(heightmap_perlin);
                        terrain_eroded = BuildTerrainModel(heightmap_eroded);
                    }
                }
                y += 40;

                if (GuiButton((Rectangle){ panelBounds.x + 80, y, 120, 30 }, "Regenerate")) {
                    // Use updated GUI values
                    GenerateLayeredNoiseHeightmap(heightmap_perlin);
                    for (int z = 0; z < MAP_SIZE; z++)
                        for (int x = 0; x < MAP_SIZE; x++)
                            heightmap_eroded[x][z] = heightmap_perlin[x][z];
                    ApplyHydroErosion(heightmap_eroded);
                    SmoothHeightmap(heightmap_eroded, 8);
                    UnloadModel(terrain_perlin);
                    UnloadModel(terrain_eroded);
                    terrain_perlin = BuildTerrainModel(heightmap_perlin);
                    terrain_eroded = BuildTerrainModel(heightmap_eroded);
                }
            }
            DrawFPS(10, 10);
            if (!showEroded) DrawText("Erosion: OFF (Perlin)", 10, 30, 18, ORANGE);
            else DrawText("Erosion: ON", 10, 30, 18, GREEN);
        EndDrawing();
    }

    UnloadModel(terrain_perlin);
    UnloadModel(terrain_eroded);
    CloseWindow();
    return 0;
}
