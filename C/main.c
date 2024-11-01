#include "raylib.h"
#include "raymath.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define MAP_SIZE 200
#define HEIGHT_SCALE 80.0f
#define NOISE_OCTAVES 8
#define NOISE_PERSISTENCE 0.5f
#define EROSION_ITERATIONS 5000

// Hash function to generate pseudo-random gradients
int Hash(int x, int y) {
    int h = x * 374761393 + y * 668265263;
    h = (h ^ (h >> 13)) * 1274126177;
    return h & 1023;
}

// Smooth interpolation function
float Smoothstep(float t) {
    return t * t * (3.0f - 2.0f * t);
}

// Dot product between distance and random gradient vectors
float Gradient(int hash, float x, float y) {
    switch (hash & 3) {
        case 0: return  x + y;
        case 1: return -x + y;
        case 2: return  x - y;
        case 3: return -x - y;
        default: return 0;
    }
}

// Perlin noise function for smoother terrain
float PerlinNoise(float x, float y) {
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

    float x1 = Lerp(n00, n10, u);
    float x2 = Lerp(n01, n11, u);
    return Lerp(x1, x2, v);
}

// Generate heightmap using layered Perlin noise
void GenerateHeightMap(float heightmap[MAP_SIZE][MAP_SIZE]) {
    for (int z = 0; z < MAP_SIZE; z++) {
        for (int x = 0; x < MAP_SIZE; x++) {
            float noiseValue = 0.0f;
            float amplitude = 1.0f;
            float frequency = 0.01f;
            for (int i = 0; i < NOISE_OCTAVES; i++) {
                noiseValue += PerlinNoise(x * frequency, z * frequency) * amplitude;
                amplitude *= NOISE_PERSISTENCE;
                frequency *= 2.0f;
            }
            noiseValue = (noiseValue + 1.0f) * 0.5f;
            heightmap[x][z] = noiseValue * HEIGHT_SCALE;
        }
    }
}

// Apply erosion to reduce peak sharpness
void ApplyErosion(float heightmap[MAP_SIZE][MAP_SIZE], int iterations) {
    for (int i = 0; i < iterations; i++) {
        int x = GetRandomValue(1, MAP_SIZE - 2);
        int z = GetRandomValue(1, MAP_SIZE - 2);
        
        float currentHeight = heightmap[x][z];
        
        for (int dz = -1; dz <= 1; dz++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dz == 0) continue;
                
                float neighborHeight = heightmap[x + dx][z + dz];
                float diff = currentHeight - neighborHeight;
                
                if (diff > 0) {
                    float erosionAmount = diff * 0.05f;
                    heightmap[x][z] -= erosionAmount;
                    heightmap[x + dx][z + dz] += erosionAmount;
                }
            }
        }
    }
}

// Linear interpolation between two colors
Color ColorLerp(Color color1, Color color2, float amount) {
    Color result;
    result.r = (unsigned char)((1.0f - amount) * color1.r + amount * color2.r);
    result.g = (unsigned char)((1.0f - amount) * color1.g + amount * color2.g);
    result.b = (unsigned char)((1.0f - amount) * color1.b + amount * color2.b);
    result.a = 255;
    return result;
}

// Define colors for different biomes
Color ICE_COLOR = { 200, 230, 255, 255 };   // Light blue for ice
Color ROCK_COLOR = { 120, 120, 120, 255 };  // Gray for rocky areas
Color WATER_COLOR = { 0, 70, 130, 255 };    // Deep blue for water
Color GRASS_COLOR = { 34, 139, 34, 255 };   // Green for grassy areas
Color SAND_COLOR = { 194, 178, 128, 255 };  // Sandy color for beaches

// Biome color based on height, simulating different biomes
Color GetBiomeColor(float height, float minHeight, float maxHeight) {
    float normalizedHeight = (height - minHeight) / (maxHeight - minHeight);

    if (normalizedHeight < 0.2f) {
        return WATER_COLOR;
    } else if (normalizedHeight < 0.3f) {
        return ColorLerp(WATER_COLOR, SAND_COLOR, (normalizedHeight - 0.2f) * 10.0f);
    } else if (normalizedHeight < 0.6f) {
        return ColorLerp(SAND_COLOR, GRASS_COLOR, (normalizedHeight - 0.3f) * 3.33f);
    } else if (normalizedHeight < 0.85f) {
        return ColorLerp(GRASS_COLOR, ROCK_COLOR, (normalizedHeight - 0.6f) * 4.0f);
    } else {
        return ColorLerp(ROCK_COLOR, ICE_COLOR, (normalizedHeight - 0.85f) * 6.67f);
    }
}

// Height-based color for height map mode
Color GetHeightMapColor(float height, float minHeight, float maxHeight) {
    float normalizedHeight = (height - minHeight) / (maxHeight - minHeight);
    return normalizedHeight < 0.5f
        ? ColorLerp(BLUE, YELLOW, normalizedHeight * 2.0f)
        : ColorLerp(YELLOW, RED, (normalizedHeight - 0.5f) * 2.0f);
}

// Generate terrain model with toggleable color mode
Model GenerateTerrainModel(float heightmap[MAP_SIZE][MAP_SIZE], bool useBiomeColors) {
    Mesh mesh = { 0 };
    mesh.vertexCount = MAP_SIZE * MAP_SIZE * 6;
    mesh.triangleCount = MAP_SIZE * MAP_SIZE * 2;

    mesh.vertices = (float *)MemAlloc(mesh.vertexCount * 3 * sizeof(float));
    mesh.colors = (unsigned char *)MemAlloc(mesh.vertexCount * 4 * sizeof(unsigned char));

    float minHeight = HEIGHT_SCALE, maxHeight = 0.0f;
    for (int z = 0; z < MAP_SIZE; z++) {
        for (int x = 0; x < MAP_SIZE; x++) {
            if (heightmap[x][z] < minHeight) minHeight = heightmap[x][z];
            if (heightmap[x][z] > maxHeight) maxHeight = heightmap[x][z];
        }
    }

    int v = 0, c = 0;
    for (int z = 0; z < MAP_SIZE - 1; z++) {
        for (int x = 0; x < MAP_SIZE - 1; x++) {
            Vector3 p1 = { x, heightmap[x][z], z };
            Vector3 p2 = { x, heightmap[x][z + 1], z + 1 };
            Vector3 p3 = { x + 1, heightmap[x + 1][z + 1], z + 1 };
            Vector3 p4 = { x + 1, heightmap[x + 1][z], z };

            Color color1 = useBiomeColors ? GetBiomeColor(heightmap[x][z], minHeight, maxHeight) : GetHeightMapColor(heightmap[x][z], minHeight, maxHeight);
            Color color2 = useBiomeColors ? GetBiomeColor(heightmap[x][z + 1], minHeight, maxHeight) : GetHeightMapColor(heightmap[x][z + 1], minHeight, maxHeight);
            Color color3 = useBiomeColors ? GetBiomeColor(heightmap[x + 1][z + 1], minHeight, maxHeight) : GetHeightMapColor(heightmap[x + 1][z + 1], minHeight, maxHeight);
            Color color4 = useBiomeColors ? GetBiomeColor(heightmap[x + 1][z], minHeight, maxHeight) : GetHeightMapColor(heightmap[x + 1][z], minHeight, maxHeight);

            mesh.vertices[v++] = p1.x; mesh.vertices[v++] = p1.y; mesh.vertices[v++] = p1.z;
            mesh.vertices[v++] = p2.x; mesh.vertices[v++] = p2.y; mesh.vertices[v++] = p2.z;
            mesh.vertices[v++] = p3.x; mesh.vertices[v++] = p3.y; mesh.vertices[v++] = p3.z;

            mesh.colors[c++] = color1.r; mesh.colors[c++] = color1.g; mesh.colors[c++] = color1.b; mesh.colors[c++] = color1.a;
            mesh.colors[c++] = color2.r; mesh.colors[c++] = color2.g; mesh.colors[c++] = color2.b; mesh.colors[c++] = color2.a;
            mesh.colors[c++] = color3.r; mesh.colors[c++] = color3.g; mesh.colors[c++] = color3.b; mesh.colors[c++] = color3.a;

            mesh.vertices[v++] = p1.x; mesh.vertices[v++] = p1.y; mesh.vertices[v++] = p1.z;
            mesh.vertices[v++] = p3.x; mesh.vertices[v++] = p3.y; mesh.vertices[v++] = p3.z;
            mesh.vertices[v++] = p4.x; mesh.vertices[v++] = p4.y; mesh.vertices[v++] = p4.z;

            mesh.colors[c++] = color1.r; mesh.colors[c++] = color1.g; mesh.colors[c++] = color1.b; mesh.colors[c++] = color1.a;
            mesh.colors[c++] = color3.r; mesh.colors[c++] = color3.g; mesh.colors[c++] = color3.b; mesh.colors[c++] = color3.a;
            mesh.colors[c++] = color4.r; mesh.colors[c++] = color4.g; mesh.colors[c++] = color4.b; mesh.colors[c++] = color4.a;
        }
    }

    UploadMesh(&mesh, true);
    Model model = LoadModelFromMesh(mesh);
    return model;
}

int main(void) {
    const int screenWidth = 1000;
    const int screenHeight = 800;

    InitWindow(screenWidth, screenHeight, "Raylib Terrain with Toggleable Color Modes");

    Camera3D camera = { 0 };
    camera.position = (Vector3){ 300.0f, 300.0f, 300.0f };
    camera.target = (Vector3){ MAP_SIZE / 2, 20.0f, MAP_SIZE / 2 };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 60.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    bool useBiomeColors = true;
    float rotationAngleY = 0.0f;
    float cameraDistance = 500.0f;

    float heightmap[MAP_SIZE][MAP_SIZE];
    GenerateHeightMap(heightmap);
    ApplyErosion(heightmap, EROSION_ITERATIONS);
    Model terrainModel = GenerateTerrainModel(heightmap, useBiomeColors);

    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_C)) {
            useBiomeColors = !useBiomeColors;
            UnloadModel(terrainModel);
            terrainModel = GenerateTerrainModel(heightmap, useBiomeColors);
        }

        if (IsKeyDown(KEY_A)) rotationAngleY += 0.01f;
        if (IsKeyDown(KEY_D)) rotationAngleY -= 0.01f;

        cameraDistance -= GetMouseWheelMove() * 5.0f;
        if (cameraDistance < 50.0f) cameraDistance = 50.0f;
        if (cameraDistance > 800.0f) cameraDistance = 800.0f;

        float camX = cameraDistance * sinf(rotationAngleY);
        float camZ = cameraDistance * cosf(rotationAngleY);
        float camY = 300.0f;

        camera.position = Vector3Add(camera.target, (Vector3){ camX, camY, camZ });

        BeginDrawing();
            ClearBackground(BLACK);
            BeginMode3D(camera);
                DrawModel(terrainModel, (Vector3){ 0, 0, 0 }, 1.0f, WHITE);
            EndMode3D();
        EndDrawing();
    }

    UnloadModel(terrainModel);
    CloseWindow();

    return 0;
}