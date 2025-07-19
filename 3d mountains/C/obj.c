#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HASH_SIZE 262144  // Must be power of two for fast modulo
#define INITIAL_CAPACITY 500000

typedef struct {
    Vector3 position;
    Vector3 normal;
    Color color;
} VertexData;

typedef struct {
    int vIdx;
    int nIdx;
    int index; // the vertex index assigned
    int occupied;
} VNHashEntry;

static VNHashEntry *hashTable = NULL;
static VertexData *vertices = NULL;
static int *indices = NULL;
static int vertexCount = 0;
static int indexCount = 0;

static int hash(int v, int n) { return ((v * 31 + n) & (HASH_SIZE - 1)); }

static int FindOrAddVertex(int vIdx, int nIdx, Vector3 *tempVerts, Vector3 *tempNorms) {
    int h = hash(vIdx, nIdx);
    while (hashTable[h].occupied) {
        if (hashTable[h].vIdx == vIdx && hashTable[h].nIdx == nIdx)
            return hashTable[h].index;
        h = (h + 1) & (HASH_SIZE - 1); // linear probing
    }

    Vector3 norm = Vector3Normalize(tempNorms[nIdx]);
    float shade = Clamp(Vector3DotProduct(norm, Vector3Normalize((Vector3){0.4f, 1.0f, 0.5f})), 0.0f, 1.0f);

    vertices[vertexCount] = (VertexData){
        .position = (Vector3){
            tempVerts[vIdx].x,
            tempVerts[vIdx].y * 1.5f,  // exaggerate height
            tempVerts[vIdx].z
        },
        
        .normal = norm,
        .color = (Color){
            60 + shade * 10,
            36 + shade * 100,
            18 + shade * 80,
            255
        }
    };
    hashTable[h] = (VNHashEntry){vIdx, nIdx, vertexCount, 1};
    return vertexCount++;
}

static Model LoadOBJAsModel(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        TraceLog(LOG_ERROR, "Failed to open OBJ file: %s", filename);
        return (Model){0};
    }

    Vector3 *tempVerts = malloc(INITIAL_CAPACITY * sizeof(Vector3));
    Vector3 *tempNorms = malloc(INITIAL_CAPACITY * sizeof(Vector3));
    vertices = malloc(INITIAL_CAPACITY * sizeof(VertexData));
    indices = malloc(INITIAL_CAPACITY * sizeof(int));
    hashTable = calloc(HASH_SIZE, sizeof(VNHashEntry));

    int vCount = 0, nCount = 0;
    char line[256];

    while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z;
            sscanf(line, "v %f %f %f", &x, &y, &z);
            tempVerts[vCount++] = (Vector3){x, y, z};
        } else if (line[0] == 'v' && line[1] == 'n') {
            float x, y, z;
            sscanf(line, "vn %f %f %f", &x, &y, &z);
            tempNorms[nCount++] = (Vector3){x, y, z};
        }
    }

    rewind(file);
    vertexCount = 0;
    indexCount = 0;

    while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'f') {
            int vi[3], ni[3];
            if (sscanf(line, "f %d//%d %d//%d %d//%d",
                       &vi[0], &ni[0], &vi[1], &ni[1], &vi[2], &ni[2]) == 6) {
                for (int i = 0; i < 3; i++) {
                    int vIdx = vi[i] - 1;
                    int nIdx = ni[i] - 1;
                    if (vIdx < 0 || nIdx < 0 || vIdx >= vCount || nIdx >= nCount) continue;
                    indices[indexCount++] = FindOrAddVertex(vIdx, nIdx, tempVerts, tempNorms);
                }
            }
        }
    }

    fclose(file);
    free(tempVerts);
    free(tempNorms);

    Mesh mesh = {0};
    mesh.vertexCount = vertexCount;
    mesh.triangleCount = indexCount / 3;
    mesh.vertices = (float *)MemAlloc(vertexCount * 3 * sizeof(float));
    mesh.normals = (float *)MemAlloc(vertexCount * 3 * sizeof(float));
    mesh.colors = (unsigned char *)MemAlloc(vertexCount * 4 * sizeof(unsigned char));
    mesh.indices = (unsigned short *)MemAlloc(indexCount * sizeof(unsigned short));

    for (int i = 0; i < vertexCount; i++) {
        mesh.vertices[i*3 + 0] = vertices[i].position.x;
        mesh.vertices[i*3 + 1] = vertices[i].position.y;
        mesh.vertices[i*3 + 2] = vertices[i].position.z;

        mesh.normals[i*3 + 0] = vertices[i].normal.x;
        mesh.normals[i*3 + 1] = vertices[i].normal.y;
        mesh.normals[i*3 + 2] = vertices[i].normal.z;

        mesh.colors[i*4 + 0] = vertices[i].color.r;
        mesh.colors[i*4 + 1] = vertices[i].color.g;
        mesh.colors[i*4 + 2] = vertices[i].color.b;
        mesh.colors[i*4 + 3] = 255;
    }

    for (int i = 0; i < indexCount; i++) mesh.indices[i] = indices[i];

    UploadMesh(&mesh, true);
    return LoadModelFromMesh(mesh);
}

int main() {
    InitWindow(1200, 800, "OBJ Viewer");
    SetTargetFPS(60);

    Model model = LoadOBJAsModel("eroded_terrain.obj");
    bool wireframe = false;

    Camera3D cam = {
        .position = { 128.0f, 120.0f, 330.0f },
        .target = { 128.0f, 0.0f, 128.0f },
        .up = { 0.0f, 1.0f, 0.0f },
        .fovy = 45.0f,
        .projection = CAMERA_PERSPECTIVE
    };

    float rotation = 0.0f;
    float distance = 300.0f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_W)) wireframe = !wireframe;
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
                if (wireframe) {
                    rlSetLineWidth(1.0f);
                    rlDrawRenderBatchActive();
                    rlBegin(RL_LINES);
                    for (int i = 0; i < indexCount; i += 3) {
                        Vector3 v0 = vertices[indices[i+0]].position;
                        Vector3 v1 = vertices[indices[i+1]].position;
                        Vector3 v2 = vertices[indices[i+2]].position;
                        rlColor4ub(255, 255, 255, 255);
                        rlVertex3f(v0.x, v0.y, v0.z);
                        rlVertex3f(v1.x, v1.y, v1.z);
                        rlVertex3f(v1.x, v1.y, v1.z);
                        rlVertex3f(v2.x, v2.y, v2.z);
                        rlVertex3f(v2.x, v2.y, v2.z);
                        rlVertex3f(v0.x, v0.y, v0.z);
                    }
                    rlEnd();
                } else {
                    DrawModel(model, (Vector3){ 0 }, 1.0f, WHITE);
                }
            EndMode3D();
            DrawFPS(10, 10);
            DrawText(TextFormat("Viewing: eroded_terrain.obj%s", wireframe ? " (Wireframe)" : ""), 10, 30, 18, GREEN);
        EndDrawing();
    }
    UnloadModel(model);
    free(vertices);
    free(indices);
    free(hashTable);
    CloseWindow();
    return 0;
}