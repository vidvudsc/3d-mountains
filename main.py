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

def fbm(x, y, octaves, persistence, lacunarity):
    noise = OpenSimplex(seed=random.randint(0, 1000000))
    value = 0
    amplitude = 1
    frequency = 1
    for _ in range(octaves):
        value += noise.noise2(x * frequency, y * frequency) * amplitude
        amplitude *= persistence
        frequency *= lacunarity
    return value

def generate_terrain():
    terrain = np.zeros((terrain_size, terrain_size), dtype=np.float32)
    moisture = np.zeros((terrain_size, terrain_size), dtype=np.float32)
    
    for y in range(terrain_size):
        for x in range(terrain_size):
            nx = x * scale
            ny = y * scale
            
            # Base terrain using fBm
            elevation = fbm(nx, ny, octaves, persistence, lacunarity)
            
            # Ridge formation
            ridge = 2 * abs(fbm(nx * 0.5, ny * 0.5, 4, 0.5, 2.0))
            elevation = elevation * (1 - ridge) + ridge * ridge
            
            # Valley formation
            valley = fbm(nx * 0.25, ny * 0.25, 2, 0.5, 2.0)
            elevation = elevation * (1 - valley * 0.3)
            
            terrain[y, x] = elevation
            
            # Generate moisture map
            moisture[y, x] = fbm(nx * 0.1, ny * 0.1, 4, 0.5, 2.0)

    # Normalize terrain and moisture
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    moisture = (moisture - moisture.min()) / (moisture.max() - moisture.min())
    
    # Apply curve to create more dramatic mountains and flatter lowlands
    terrain = np.power(terrain, 1.2) * 120

    # Simulate erosion
    terrain = apply_erosion(terrain)

    return terrain, moisture

def apply_erosion(terrain):
    iterations = 50000
    rain_amount = 0.01
    evaporation_rate = 0.5
    capacity = 4
    erosion_rate = 0.3

    water = np.zeros_like(terrain)
    sediment = np.zeros_like(terrain)

    for _ in range(iterations):
        x, y = random.randint(1, terrain_size-2), random.randint(1, terrain_size-2)
        water[y, x] += rain_amount

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            height_diff = (terrain[y, x] + water[y, x]) - (terrain[ny, nx] + water[ny, nx])
            if height_diff > 0:
                transfer = min(height_diff / 4, water[y, x])
                water[y, x] -= transfer
                water[ny, nx] += transfer

                sediment_capacity = capacity * height_diff
                if sediment[y, x] > sediment_capacity:
                    deposit = sediment[y, x] - sediment_capacity
                    sediment[y, x] -= deposit
                    terrain[y, x] += deposit
                else:
                    eroded = min(erosion_rate * height_diff, terrain[y, x])
                    terrain[y, x] -= eroded
                    sediment[y, x] += eroded

        water[y, x] *= (1 - evaporation_rate)
        sediment[y, x] *= (1 - evaporation_rate)

    return terrain

def get_biome_color(height, moisture, slope):
    # Base colors (more varied and natural)
    SNOW = np.array([0.95, 0.95, 0.97])
    ROCK = np.array([0.5, 0.48, 0.45])
    TUNDRA = np.array([0.65, 0.7, 0.6])
    BARE = np.array([0.7, 0.6, 0.5])
    SCORCHED = np.array([0.5, 0.4, 0.3])
    TAIGA = np.array([0.2, 0.35, 0.25])
    SHRUBLAND = np.array([0.4, 0.45, 0.3])
    TEMPERATE_DESERT = np.array([0.85, 0.75, 0.6])
    TEMPERATE_RAIN_FOREST = np.array([0.1, 0.3, 0.15])
    TEMPERATE_DECIDUOUS_FOREST = np.array([0.25, 0.4, 0.2])
    GRASSLAND = np.array([0.55, 0.6, 0.35])
    TROPICAL_RAIN_FOREST = np.array([0.1, 0.35, 0.1])
    TROPICAL_SEASONAL_FOREST = np.array([0.2, 0.45, 0.2])
    SUBTROPICAL_DESERT = np.array([0.8, 0.7, 0.5])

    height = max(0, min(1, height / 120))
    moisture = max(0, min(1, moisture))
    slope = max(0, min(1, slope))

    if height > 0.8:
        return SNOW
    elif height > 0.7:
        if moisture < 0.1:
            return SCORCHED
        elif moisture < 0.2:
            return BARE
        elif moisture < 0.5:
            return TUNDRA
        else:
            return SNOW
    elif height > 0.6:
        if moisture < 0.33:
            return TEMPERATE_DESERT
        elif moisture < 0.66:
            return SHRUBLAND
        else:
            return TAIGA
    elif height > 0.3:
        if moisture < 0.16:
            return TEMPERATE_DESERT
        elif moisture < 0.50:
            return GRASSLAND
        elif moisture < 0.83:
            return TEMPERATE_DECIDUOUS_FOREST
        else:
            return TEMPERATE_RAIN_FOREST
    else:
        if moisture < 0.16:
            return SUBTROPICAL_DESERT
        elif moisture < 0.33:
            return GRASSLAND
        elif moisture < 0.66:
            return TROPICAL_SEASONAL_FOREST
        else:
            return TROPICAL_RAIN_FOREST

    # Blend with ROCK color based on slope
    return color * (1 - slope * 0.5) + ROCK * (slope * 0.5)

def calculate_normal(v1, v2, v3, v4):
    u = np.array(v2) - np.array(v1)
    v = np.array(v4) - np.array(v1)
    normal = np.cross(u, v)
    return normal / np.linalg.norm(normal)

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
        slope = 1 - normal[1]  # Use the y-component of the normal to estimate slope
        
        # Calculate colors for each vertex
        c1 = get_biome_color(y1, moisture[i,j], slope)
        c2 = get_biome_color(y2, moisture[i+1,j], slope)
        c3 = get_biome_color(y3, moisture[i+1,j+1], slope)
        c4 = get_biome_color(y4, moisture[i,j+1], slope)
        
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