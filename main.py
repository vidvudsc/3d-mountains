import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from opensimplex import OpenSimplex
import random

# Initialize Pygame and OpenGL
pygame.init()
width, height = 1024, 768
pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
clock = pygame.time.Clock()

# Terrain settings
terrain_size = 512
scale = 0.003
octaves = 8
persistence = 0.5
lacunarity = 2.0

# Camera settings
camera_distance = 750
camera_angle_x = 30
camera_angle_y = 0

def generate_terrain():
    seed = random.randint(0, 1000000)
    noise = OpenSimplex(seed=seed)
    
    def noise_at(nx, ny, octaves, persistence, lacunarity):
        value = 0
        amplitude = 1
        frequency = 1
        for _ in range(octaves):
            value += noise.noise2(nx * frequency, ny * frequency) * amplitude
            amplitude *= persistence
            frequency *= lacunarity
        return value

    terrain = np.zeros((terrain_size, terrain_size), dtype=np.float32)
    moisture = np.zeros((terrain_size, terrain_size), dtype=np.float32)
    for y in range(terrain_size):
        for x in range(terrain_size):
            nx = x * scale
            ny = y * scale
            
            # Base terrain (creates general landscape)
            base = noise_at(nx, ny, octaves, persistence, lacunarity)
            
            # Mountain ranges (creates areas of higher elevation)
            mountain_range = noise_at(nx * 0.2, ny * 0.2, 4, 0.5, 2.0)
            mountain_range = np.maximum(mountain_range, 0)  # Only positive values
            
            # Combine base and mountain range
            combined = base * 0.7 + mountain_range * 0.3
            
            # Add some roughness
            roughness = noise_at(nx * 4, ny * 4, 2, 0.5, 2.0) * 0.03
            
            terrain[y, x] = combined + roughness
            
            # Generate moisture map
            moisture[y, x] = noise_at(nx * 0.1, ny * 0.1, 4, 0.5, 2.0)

    # Normalize terrain and moisture
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    moisture = (moisture - moisture.min()) / (moisture.max() - moisture.min())
    
    # Apply curve to create more dramatic mountains and flatter lowlands
    terrain = np.power(terrain, 1.2) * 120

    # Simulate erosion
    for _ in range(3):
        terrain = apply_erosion(terrain)

    return terrain, moisture

def apply_erosion(terrain):
    eroded = terrain.copy()
    for _ in range(50000):
        x, y = random.randint(1, terrain_size-2), random.randint(1, terrain_size-2)
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            diff = eroded[y, x] - eroded[ny, nx]
            if diff > 0:
                amount = min(diff * 0.1, 0.5)
                eroded[y, x] -= amount
                eroded[ny, nx] += amount
    return eroded

def calculate_normal(v1, v2, v3, v4):
    u = np.array(v2) - np.array(v1)
    v = np.array(v4) - np.array(v1)
    normal = np.cross(u, v)
    return normal / np.linalg.norm(normal)

def get_biome_color(height, moisture):
    # Base colors (even more muted and natural)
    SNOW = np.array([0.95, 0.95, 0.97])
    ROCK = np.array([0.5, 0.48, 0.45])
    FOREST = np.array([0.2, 0.35, 0.1])
    GRASSLAND = np.array([0.3, 0.4, 0.2])
    SAVANNA = np.array([0.5, 0.45, 0.3])
    DESERT = np.array([0.7, 0.6, 0.5])

    height = max(0, min(1, height))  # Clamp height to [0, 1]
    moisture = max(0, min(1, moisture))  # Clamp moisture to [0, 1]

    if height > 0.85:
        return SNOW
    elif height > 0.65:
        t = (height - 0.65) / 0.2
        return ROCK * t + FOREST * (1 - t)
    elif height > 0.35:
        if moisture > 0.5:
            return FOREST
        else:
            t = (moisture - 0.3) / 0.2
            return FOREST * t + GRASSLAND * (1 - t)
    else:
        if moisture > 0.6:
            return GRASSLAND
        elif moisture > 0.3:
            t = (moisture - 0.3) / 0.3
            return GRASSLAND * t + SAVANNA * (1 - t)
        else:
            t = moisture / 0.3
            return SAVANNA * t + DESERT * (1 - t)

def apply_simple_lighting(color, normal):
    light_dir = np.array([0.5, 1, 0.5])
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    ambient = 0.3
    diffuse = max(0, np.dot(normal, light_dir))
    
    return color * (ambient + diffuse * 0.7)

print("Generating terrain...")
terrain, moisture = generate_terrain()
print("Terrain generated.")

# Create vertex and color arrays
print("Creating vertex and color arrays...")
vertices = []
colors = []
for i in range(terrain_size - 1):
    for j in range(terrain_size - 1):
        x, z = i - terrain_size // 2, j - terrain_size // 2
        y1, y2, y3, y4 = terrain[i,j], terrain[i+1,j], terrain[i+1,j+1], terrain[i,j+1]
        v1 = (x, y1, -z)
        v2 = (x + 1, y2, -z)
        v3 = (x + 1, y3, -(z + 1))
        v4 = (x, y4, -(z + 1))
        
        normal = calculate_normal(v1, v2, v3, v4)
        
        # Calculate colors for each vertex
        c1 = get_biome_color(y1/120, moisture[i,j])
        c2 = get_biome_color(y2/120, moisture[i+1,j])
        c3 = get_biome_color(y3/120, moisture[i+1,j+1])
        c4 = get_biome_color(y4/120, moisture[i,j+1])
        
        # Apply simple lighting
        c1 = apply_simple_lighting(c1, normal)
        c2 = apply_simple_lighting(c2, normal)
        c3 = apply_simple_lighting(c3, normal)
        c4 = apply_simple_lighting(c4, normal)
        
        vertices.extend([v1, v2, v3, v4])
        colors.extend([c1, c2, c3, c4])

vertices = np.array(vertices, dtype=np.float32)
colors = np.array(colors, dtype=np.float32)
print("Arrays created.")

# Set up OpenGL
glEnable(GL_DEPTH_TEST)
glEnable(GL_CULL_FACE)
glCullFace(GL_BACK)

# Set up fog for atmospheric effect
glEnable(GL_FOG)
glFogi(GL_FOG_MODE, GL_LINEAR)
glFogfv(GL_FOG_COLOR, (0.7, 0.7, 0.8, 1))
glFogf(GL_FOG_START, 500)
glFogf(GL_FOG_END, 1500)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEWHEEL:
            camera_distance -= event.y * 20
            camera_distance = max(100, min(1500, camera_distance))

    keys = pygame.key.get_pressed()
    if keys[pygame.K_a]:
        camera_angle_y -= 2
    if keys[pygame.K_d]:
        camera_angle_y += 2

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # Set up camera
    gluPerspective(45, (width / height), 0.1, 3000.0)
    glTranslatef(0, 0, -camera_distance)
    glRotatef(camera_angle_x, 1, 0, 0)
    glRotatef(camera_angle_y, 0, 1, 0)

    # Render terrain
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, vertices)
    glColorPointer(3, GL_FLOAT, 0, colors)
    glDrawArrays(GL_QUADS, 0, len(vertices))
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_COLOR_ARRAY)

    pygame.display.flip()
    clock.tick(60)
    print(f"FPS: {clock.get_fps():.2f}")

pygame.quit()