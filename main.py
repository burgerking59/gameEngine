import glfw
import glfw.GLFW as GLFW_CONSTANTS

from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
from PIL import Image

ENTITY_TYPE = {
    "CUBE": 0,
    "POINTLIGHT": 1,
    "DONUT": 2,
}

WIDTH = 1280
HEIGHT = 960

UNIFORM_TYPE = {
    "MODEL": 0,
    "VIEW": 1,
    "PROJECTION": 2,
    "CAMERA_POS": 3,
    "LIGHT_COLOR": 4,
    "LIGHT_POS": 5,
    "LIGHT_STRENGTH": 6,
}

GLOBAL_X = np.array([1,0,0], dtype=np.float32)
GLOBAL_Y = np.array([0,1,0], dtype=np.float32)
GLOBAL_Z = np.array([0,0,1], dtype=np.float32)

def loadMesh(filename: str) -> list[float]:
    """
        Load a mesh from an obj file.

        Parameters:

            filename: the filename.
        
        Returns:

            The loaded data, in a flattened format.
    """
    v = []
    vt = []
    vn = []
    vertices = []
    with open(filename, "r") as file:
        line = file.readline()
        while line:
            words = line.split(" ")
            match words[0]:
                case "v":
                    v.append(read_vertex_data(words))

                case "vt":
                    vt.append(read_texcoord_data(words))
                
                case "vn":
                    vn.append(read_normal_data(words))
            
                case "f":
                    read_face_data(words, v, vt, vn, vertices)
            line = file.readline()
    return vertices

def read_vertex_data(words: list[str]) -> list[float]:
    """
        Returns a vertex description.
    """

    return [
        float(words[1]),
        float(words[2]),
        float(words[3])
    ]

def read_texcoord_data(words: list[str]) -> list[float]:
    """
        Returns a texture coordinate description.
    """

    return [
        float(words[1]),
        float(words[2])
    ]
    
def read_normal_data(words: list[str]) -> list[float]:
    """
        Returns a normal vector description.
    """

    return [
        float(words[1]),
        float(words[2]),
        float(words[3])
    ]

def read_face_data(
    words: list[str], 
    v: list[list[float]], vt: list[list[float]], 
    vn: list[list[float]], vertices: list[float]) -> None:
    """
        Reads an edgetable and makes a face from it.
    """

    triangleCount = len(words) - 3

    for i in range(triangleCount):

        make_corner(words[1], v, vt, vn, vertices)
        make_corner(words[2 + i], v, vt, vn, vertices)
        make_corner(words[3 + i], v, vt, vn, vertices)
    
def make_corner(corner_description: str, 
    v: list[list[float]], vt: list[list[float]], 
    vn: list[list[float]], vertices: list[float]) -> None:
    """
        Composes a flattened description of a vertex.
    """

    v_vt_vn = corner_description.split("/")
    
    for element in v[int(v_vt_vn[0]) - 1]:
        vertices.append(element)
    for element in vt[int(v_vt_vn[1]) - 1]:
        vertices.append(element)
    for element in vn[int(v_vt_vn[2]) - 1]:
        vertices.append(element)

def createShader(vertex_filepath, fragment_filepath):
    with open(vertex_filepath,'r') as f:
        vertex_src = f.readlines()

    with open(fragment_filepath,'r') as f:
        fragment_src = f.readlines()
    
    shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                            compileShader(fragment_src, GL_FRAGMENT_SHADER))
    
    return shader

class App:
    """
    The control class
    """


    def __init__(self):
        glfw.init()
        glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MAJOR,3)
        glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MINOR,3)
        glfw.window_hint(
            GLFW_CONSTANTS.GLFW_OPENGL_PROFILE, 
            GLFW_CONSTANTS.GLFW_OPENGL_CORE_PROFILE)
        glfw.window_hint(GLFW_CONSTANTS.GLFW_OPENGL_FORWARD_COMPAT, GLFW_CONSTANTS.GLFW_TRUE)
        #for uncapped framerate
        glfw.window_hint(GLFW_CONSTANTS.GLFW_DOUBLEBUFFER,GL_FALSE) 
        self.window = glfw.create_window(
            WIDTH, HEIGHT, "Title", None, None)
        glfw.make_context_current(self.window)

        self._set_up_timer()
        self._set_up_input_systems()

        
        self.renderer = GraphicsEngine()
        self.scene = Scene(self.renderer.shaders)
        
        self.mainLoop()
    def mainLoop(self):
        running = True
        while running:
            if glfw.window_should_close(self.window) \
                or self._keys.get(GLFW_CONSTANTS.GLFW_KEY_ESCAPE, False):
                running = False
            
            self.handle_keys()
            self.handle_mouse()

            glfw.poll_events()

            self.scene.update(self.frametime / 16.67, self.renderer.shaders)
            
            self.renderer.render(self.scene.player, self.scene.entities)

            #timing
            self._calculate_framerate()
        
    def _set_up_timer(self) -> None:
        """
            Initialize the variables used by the framerate
            timer.
        """

        self.last_time = glfw.get_time()
        self.current_time = 0
        self.frames_rendered = 0
        self.frametime = 0.0
    
    def _set_up_input_systems(self) -> None:
        """
            Configure the mouse and keyboard
        """

        glfw.set_input_mode(
            self.window, 
            GLFW_CONSTANTS.GLFW_CURSOR, 
            GLFW_CONSTANTS.GLFW_CURSOR_HIDDEN
        )

        self._keys = {}
        glfw.set_key_callback(self.window, self._key_callback)

    def _key_callback(self, window, key, scancode, action, mods) -> None:
        """
            Handle a key event.

            Parameters:

                window: the window on which the keypress occurred.

                key: the key which was pressed

                scancode: scancode of the key

                action: action of the key event

                mods: modifiers applied to the event
        """

        state = False
        match action:
            case GLFW_CONSTANTS.GLFW_PRESS:
                state = True
            case GLFW_CONSTANTS.GLFW_RELEASE:
                state = False
            case _:
                return

        self._keys[key] = state

    def handle_keys(self):
        rate = 0.005*self.frametime
        d_pos = np.zeros(3, dtype=np.float32)
        
        if self._keys.get(GLFW_CONSTANTS.GLFW_KEY_W, False):
            d_pos += GLOBAL_X
        if self._keys.get(GLFW_CONSTANTS.GLFW_KEY_A, False):
            d_pos -= GLOBAL_Y
        if self._keys.get(GLFW_CONSTANTS.GLFW_KEY_S, False):
            d_pos -= GLOBAL_X
        if self._keys.get(GLFW_CONSTANTS.GLFW_KEY_D, False):
            d_pos += GLOBAL_Y

        length = pyrr.vector.length(d_pos)
        if abs(length) < 0.00001:
            return

        d_pos = rate * d_pos / length
        self.scene.move_player(d_pos)

    def handle_mouse(self):
        (x,y) = glfw.get_cursor_pos(self.window)
        d_eulers = 0.02 * ((WIDTH / 2) - x) * GLOBAL_Z
        d_eulers += 0.02 * ((HEIGHT / 2) - y) * GLOBAL_Y
        self.scene.spin_player(d_eulers)
        glfw.set_cursor_pos(self.window, WIDTH / 2, HEIGHT / 2)

    def _calculate_framerate(self) -> None:
        """
            Calculate the framerate and frametime,
            and update the window title.
        """

        self.current_time = glfw.get_time()
        delta = self.current_time - self.last_time
        if (delta >= 1):
            framerate = max(1,int(self.frames_rendered/delta))
            glfw.set_window_title(self.window, f"Running at {framerate} fps.")
            self.last_time = self.current_time
            self.frames_rendered = -1
            self.frametime = float(1000.0 / max(1,framerate))
        self.frames_rendered += 1

class Entity:
    """
        A basic object in the world, with a position and rotation.
    """
    __slots__ = ("position", "eulers")

    def __init__(self, position: list[float], eulers: list[float]):
        """
            Initialize the entity.

            Parameters:

                position: the position of the entity.

                eulers: the rotation of the entity
                        about each axis.
        """

        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)

    def update(self, dt: float) -> None:
        """
            Update the object, this is meant to be implemented by
            objects extending this class.

            Parameters:

                dt: framerate correction factor.
        """

        pass

    def get_model_transform(self) -> np.ndarray:
        """
            Returns the entity's model to world
            transformation matrix.
        """

        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)

        model_transform = pyrr.matrix44.multiply(
            m1=model_transform, 
            m2=pyrr.matrix44.create_from_axis_rotation(
                axis = GLOBAL_Z,
                theta = np.radians(self.eulers[2]), 
                dtype = np.float32
            )
        )

        return pyrr.matrix44.multiply(
            m1=model_transform, 
            m2=pyrr.matrix44.create_from_translation(
                vec=np.array(self.position),dtype=np.float32
            )
        )

class Cube(Entity):
    """
        A basic object in the world, with a position and rotation.
    """
    __slots__ = tuple([])

    def __init__(self, position: list[float], eulers: list[float]):
        """
            Initialize the cube.

            Parameters:

                position: the position of the entity.

                eulers: the rotation of the entity
                        about each axis.
        """

        super().__init__(position, eulers)
    
    def update(self, dt: float) -> None:
        """
            Update the cube.

            Parameters:

                dt: framerate correction factor.
        """
        return

class Donut(Entity):
    """
        A basic object in the world, with a position and rotation.
    """
    __slots__ = tuple([])

    def __init__(self, position: list[float], eulers: list[float]):
        """
            Initialize the cube.

            Parameters:

                position: the position of the entity.

                eulers: the rotation of the entity
                        about each axis.
        """

        super().__init__(position, eulers)
    
    def update(self, dt: float) -> None:
        """
            Update the cube.

            Parameters:

                dt: framerate correction factor.
        """
        pass

class PointLight(Entity):
    """
        A simple pointlight.
    """
    __slots__ = ("color", "strength")


    def __init__(
        self, position: list[float], 
        color: list[float], strength: float):
        """
            Initialize the light.

            Parameters:

                position: position of the light.

                color: (r,g,b) color of the light.

                strength: strength of the light.
        """

        super().__init__(position, eulers = [0,0,0])
        self.color = np.array(color, dtype=np.float32)
        self.strength = strength

class Camera(Entity):
    """
        A first person camera.
    """
    __slots__ = ("forwards", "right", "up")


    def __init__(self, position: list[float], shaders):
        """
            Initialize the camera.

            Parameters:

                position: the camera's position
        """

        super().__init__(position, eulers = [0,0,0])
        self.update(0, shaders)
    
    def update(self, dt: float, shaders) -> None:
        """
            Update the camera.

            Parameters:

                dt: framerate correction factor
        """

        theta = self.eulers[2]
        phi = self.eulers[1]

        self.forwards = np.array(
            [
                np.cos(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)),
                np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)),
                np.sin(np.deg2rad(phi))
            ],
            dtype = np.float32
        )

        self.right = np.cross(self.forwards, GLOBAL_Z)

        self.up = np.cross(self.right, self.forwards)

        lookat_matrix = pyrr.matrix44.create_look_at(self.position, self.position + self.forwards, self.up,dtype=np.float32)
        for shader in shaders:
            glUseProgram(shader)
            glUniformMatrix4fv(glGetUniformLocation(shader,"view"),1,GL_FALSE,lookat_matrix)
            glUniform3fv(glGetUniformLocation(shader,"cameraPos"),1,self.position)

    def get_view_transform(self) -> np.ndarray:
        """
            Returns the camera's world to view
            transformation matrix.
        """

        return pyrr.matrix44.create_look_at(
            eye = self.position,
            target = self.position + self.forwards,
            up = self.up, dtype = np.float32)
    
    def move(self, d_pos) -> None:
        """
            Move by the given amount in the (forwards, right, up) vectors.
        """

        self.position += d_pos[0] * self.forwards \
                        + d_pos[1] * self.right \
                        + d_pos[2] * self.up
    
        #hard coding the z constraint
        self.position[2] = 2
    
    def spin(self, d_eulers) -> None:
        """
            Spin the camera by the given amount about the (x, y, z) axes.
        """

        self.eulers += d_eulers

        self.eulers[0] %= 360
        self.eulers[1] = min(89, max(-89, self.eulers[1]))
        self.eulers[2] %= 360

class GraphicsEngine:
    """
        Draws entities and stuff.
    """
    __slots__ = (
        "meshes", "materials", "shader",
        "skyboxShader", "shaders", "skyBoxTexture", 
        "skyBox", "uniform_locations", "light_locations")


    def __init__(self):
        """
            Initializes the rendering system.
        """

        self._set_up_opengl()

        self._create_assets()

        self._set_onetime_uniforms()

        self._get_uniform_locations()
    
    def _set_up_opengl(self) -> None:
        """
            Configure any desired OpenGL options
        """

        glClearColor(0.0, 0.0, 0.0, 1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
    
    def _create_assets(self) -> None:
        """
            Create all of the assets needed for drawing.
        """
        #self.colliders: dict[int, Collider] = {
            #ENTITY_TYPE["CUBE"]: Collider("assets/Cube.obj"),
            #ENTITY_TYPE["DONUT"]: Collider("assets/Cube.obj"),
        #}
        self.meshes: dict[int, Mesh] = {
            ENTITY_TYPE["CUBE"]: Mesh("assets/Cube.obj"),
            ENTITY_TYPE["DONUT"]: Mesh("assets/blueCube.obj"),
        }
        self.materials: dict[int, Material] = {
            ENTITY_TYPE["CUBE"] : Material("assets/wood.jpeg"),
            ENTITY_TYPE["DONUT"] : Material("assets/donut.png"),
        }
        
        self.shader = createShader(
            vertex_filepath = "shaders/vertex.txt", 
            fragment_filepath = "shaders/fragment.txt")
        
        self.skyboxShader = createShader(
            vertex_filepath = "shaders/vertex-skybox.txt", 
            fragment_filepath = "shaders/fragment-skybox.txt")
        self.shaders = [self.shader, self.skyboxShader]
        
        self.skyBoxTexture = CubeMapMaterial("gfx/sky")
        skyBoxModel = CubeMapModel(self.skyboxShader, 40,40,40,1,1,1, self.skyBoxTexture)
        self.skyBox = skyBox(skyBoxModel)
    
    def _set_onetime_uniforms(self) -> None:
        """
            Some shader data only needs to be set once.
        """

        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy = 45, aspect = 640/480, 
            near = 0.1, far = 50, dtype=np.float32
        )
        

        glUseProgram(self.shader)
        glUniform1i(glGetUniformLocation(self.shader, "imageTexture"), 0)
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader,"projection"),
            1, GL_FALSE, projection_transform
        )

        glUseProgram(self.skyboxShader)
        glUniform1i(glGetUniformLocation(self.skyboxShader, "skyBox"), 0)
        glUniformMatrix4fv(glGetUniformLocation(self.skyboxShader,"projection"),
            1,GL_FALSE,projection_transform
        )

        
    
    def _get_uniform_locations(self) -> None:
        """
            Query and store the locations of shader uniforms
        """

        glUseProgram(self.skyboxShader)
        self.uniform_locations: dict[int, int] = {
            UNIFORM_TYPE["CAMERA_POS"]: glGetUniformLocation(
                self.skyboxShader, "cameraPosition"),
            UNIFORM_TYPE["MODEL"]: glGetUniformLocation(self.skyboxShader, "model"),
            UNIFORM_TYPE["VIEW"]: glGetUniformLocation(self.skyboxShader, "view"),
        }

        self.light_locations: dict[int, list[int]] = {
            UNIFORM_TYPE["LIGHT_COLOR"]: [
                glGetUniformLocation(self.skyboxShader, f"Lights[{i}].color")
                for i in range(8)
            ],
            UNIFORM_TYPE["LIGHT_POS"]: [
                glGetUniformLocation(self.skyboxShader, f"Lights[{i}].position")
                for i in range(8)
            ],
            UNIFORM_TYPE["LIGHT_STRENGTH"]: [
                glGetUniformLocation(self.skyboxShader, f"Lights[{i}].strength")
                for i in range(8)
            ],
        }

        glUseProgram(self.shader)
        self.uniform_locations: dict[int, int] = {
            UNIFORM_TYPE["CAMERA_POS"]: glGetUniformLocation(
                self.shader, "cameraPosition"),
            UNIFORM_TYPE["MODEL"]: glGetUniformLocation(self.shader, "model"),
            UNIFORM_TYPE["VIEW"]: glGetUniformLocation(self.shader, "view"),
        }

        self.light_locations: dict[int, list[int]] = {
            UNIFORM_TYPE["LIGHT_COLOR"]: [
                glGetUniformLocation(self.shader, f"Lights[{i}].color")
                for i in range(8)
            ],
            UNIFORM_TYPE["LIGHT_POS"]: [
                glGetUniformLocation(self.shader, f"Lights[{i}].position")
                for i in range(8)
            ],
            UNIFORM_TYPE["LIGHT_STRENGTH"]: [
                glGetUniformLocation(self.shader, f"Lights[{i}].strength")
                for i in range(8)
            ],
        }

        

    def render(self, 
        camera: Camera, renderables: dict[int, list[Entity]]) -> None:
        """
            Draw everything.

            Parameters:

                camera: the scene's camera

                renderables: all the entities to draw
        """

        #refresh screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDisable(GL_CULL_FACE)
        self.skyBox.draw(camera.position)
        glEnable(GL_CULL_FACE)
        
        glUseProgram(self.shader)

        glUniformMatrix4fv(
            self.uniform_locations[UNIFORM_TYPE["VIEW"]], 
            1, GL_FALSE, camera.get_view_transform())
        
        for i in range(len(renderables[ENTITY_TYPE["POINTLIGHT"]])):

            light: PointLight = renderables[ENTITY_TYPE["POINTLIGHT"]][i]

            glUniform3fv(
                self.light_locations[UNIFORM_TYPE["LIGHT_POS"]][i], 
                1, light.position)
            glUniform3fv(
                self.light_locations[UNIFORM_TYPE["LIGHT_COLOR"]][i], 
                1, light.color)
            glUniform1f(
                self.light_locations[UNIFORM_TYPE["LIGHT_STRENGTH"]][i], 
                light.strength)

        glUniform3fv(self.uniform_locations[UNIFORM_TYPE["CAMERA_POS"]],
            1, camera.position)
        
        
        for entity_type, entities in renderables.items():

            if entity_type not in self.meshes:
                continue

            mesh = self.meshes[entity_type]
            material = self.materials[entity_type]
            mesh.arm_for_drawing()
            material.use()
            for entity in entities:
                glUniformMatrix4fv(
                    self.uniform_locations[UNIFORM_TYPE["MODEL"]],
                    1, GL_FALSE, entity.get_model_transform())
                mesh.draw()
        
        glFlush()

    def destroy(self) -> None:
        """ free any allocated memory """

        for mesh in self.meshes.values():
            mesh.destroy()
        for material in self.materials.values():
            material.destroy()
        glDeleteProgram(self.shader)
        glDeleteProgram(self.skyboxShader)
        self.skyBox.destroy()

class Scene:
    """
        Manages all objects and coordinates their interactions.
    """
    __slots__ = ("entities", "player")

    def __init__(self, shaders):
        """
            Initialize the scene.
        """
        self.entities: dict[int, list[Entity]] = {
            ENTITY_TYPE["DONUT"]: [
                Donut(position = [6,0,0], eulers = [0,0,0]),
                Donut(position = [6,6,0], eulers = [0,0,0]),
                Donut(position = [6,-6,6], eulers = [0,0,0]),
                Donut(position = [12,-6,0], eulers = [0,0,0]),
                Donut(position = [12,0,0], eulers = [0,0,0]),
                Donut(position = [12,0,6], eulers = [0,0,0]),
                Donut(position = [6,0,6], eulers = [0,0,0]),
            ],

            ENTITY_TYPE["CUBE"]: [
                Cube(position=[3.5,1,-0.5], eulers=[0,0,0]),
                Cube(position=[4,1,0], eulers=[0,0,0]),

                Cube(position=[1,1,-1], eulers=[0,0,0]),
                Cube(position=[3,1,-1], eulers=[0,0,0]),
                Cube(position=[5,1,-1], eulers=[0,0,0]),
                Cube(position=[7,1,-1], eulers=[0,0,0]),
                Cube(position=[9,1,-1], eulers=[0,0,0]),

                Cube(position=[1,3,-1], eulers=[0,0,0]),
                Cube(position=[1,5,-1], eulers=[0,0,0]),
                Cube(position=[1,7,-1], eulers=[0,0,0]),
                Cube(position=[1,9,-1], eulers=[0,0,0]),

                Cube(position=[3,3,-1], eulers=[0,0,0]),
                Cube(position=[3,5,-1], eulers=[0,0,0]),
                Cube(position=[3,7,-1], eulers=[0,0,0]),
                Cube(position=[3,9,-1], eulers=[0,0,0]),

                Cube(position=[5,3,-1], eulers=[0,0,0]),
                Cube(position=[5,5,-1], eulers=[0,0,0]),
                Cube(position=[5,7,-1], eulers=[0,0,0]),
                Cube(position=[5,9,-1], eulers=[0,0,0]),

                Cube(position=[7,3,-1], eulers=[0,0,0]),
                Cube(position=[7,5,-1], eulers=[0,0,0]),
                Cube(position=[7,7,-1], eulers=[0,0,0]),
                Cube(position=[7,9,-1], eulers=[0,0,0]),

                Cube(position=[9,3,-1], eulers=[0,0,0]),
                Cube(position=[9,5,-1], eulers=[0,0,0]),
                Cube(position=[9,7,-1], eulers=[0,0,0]),
                Cube(position=[9,9,-1], eulers=[0,0,0]),
            ],
            
            ENTITY_TYPE["POINTLIGHT"]: [
                PointLight(
                    position = [
                        3, 
                        -3, 
                        4],
                    color = [
                        5, 
                        15, 
                        15],
                    strength = 4)
            ],
        }

        self.player = Camera(
            position = [0,0,2], shaders = shaders
        )

    def update(self, dt: float, shaders):
        """
            Update all objects in the scene.

            Parameters:

                dt: framerate correction factor
        """
        for entities in self.entities.values():
            for entity in entities:
                entity.update(dt)

        self.player.update(dt, shaders)

    def move_player(self, d_pos):
        """
            move the player by the given amount in the 
            (forwards, right, up) vectors.
        """

        self.player.move(d_pos)
    
    def spin_player(self, eulers) -> None:
        """
            spin the player by the given amount
            around the (x,y,z) axes
        """

        self.player.spin(eulers)

    

class Mesh:
    """
        A mesh that can represent an obj model.
    """

    def __init__(self, filename: str):
        """
            Initialize the mesh.
        """

        # x, y, z, s, t, nx, ny, nz
        vertices = loadMesh(filename)
        self.vertex_count = len(vertices)//8
        vertices = np.array(vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        #Vertices
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        #position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        #texture
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        #normal
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))
    
    def arm_for_drawing(self) -> None:
        """
            Arm the triangle for drawing.
        """
        glBindVertexArray(self.vao)
    
    def draw(self) -> None:
        """
            Draw the triangle.
        """

        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)

    def destroy(self) -> None:
        """
            Free any allocated memory.
        """
        
        glDeleteVertexArrays(1,(self.vao,))
        glDeleteBuffers(1,(self.vbo,))

class Material:
    """
        A basic texture.
    """
    __slots__ = ("texture",)

    
    def __init__(self, filepath: str):
        """
            Initialize and load the texture.

            Parameters:

                filepath: path to the image file.
        """

        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        with Image.open(filepath, mode = "r") as img:
            image_width,image_height = img.size
            img = img.convert("RGBA")
            img_data = bytes(img.tobytes())
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)

        glGenerateMipmap(GL_TEXTURE_2D)

    def use(self) -> None:
        """
            Arm the texture for drawing.
        """

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D,self.texture)

    def destroy(self) -> None:
        """
            Free the texture.
        """

        glDeleteTextures(1, (self.texture,))

class CubeMapMaterial:
    def __init__(self, filepath):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.texture)

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        #load textures
        with Image.open("assets/left.jpg", mode = "r") as img:
            image_width,image_height = img.size
            img = img.convert("RGBA")
            img_data = bytes(img.tobytes())
            glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,0,GL_RGBA8,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)

        with Image.open("assets/right.jpg", mode = "r") as img:
            img = img.rotate(180)
            image_width,image_height = img.size
            img = img.convert("RGBA")
            img_data = bytes(img.tobytes())
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y,0,GL_RGBA8,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)

        with Image.open("assets/top.jpg", mode = "r") as img:
            img = img.rotate(90)
            image_width,image_height = img.size
            img = img.convert("RGBA")
            img_data = bytes(img.tobytes())
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z,0,GL_RGBA8,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)

        with Image.open("assets/bottom.jpg", mode = "r") as img:
            img = img.rotate(90)
            image_width,image_height = img.size
            img = img.convert("RGBA")
            img_data = bytes(img.tobytes())
            glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,0,GL_RGBA8,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)

        with Image.open("assets/back.jpg", mode = "r") as img:
            img = img.rotate(-90)
            image_width,image_height = img.size
            img = img.convert("RGBA")
            img_data = bytes(img.tobytes())
            glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X,0,GL_RGBA8,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)

        with Image.open("assets/front.jpg", mode = "r") as img:
            img = img.rotate(90)
            image_width,image_height = img.size
            img = img.convert("RGBA")
            img_data = bytes(img.tobytes())
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X,0,GL_RGBA8,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)


    def use(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP,self.texture)

    def destroy(self):
        glDeleteTextures(1, (self.texture,))

class CubeMapModel:
    def __init__(self, shader, l, w, h, r, g, b, material):
        self.material = material
        self.shader = shader
        glUseProgram(shader)
        # x, y, z, r, g, b
        self.vertices = (
                -l/2, -w/2, -h/2, r, g, b,
                -l/2,  w/2, -h/2, r, g, b,
                 l/2,  w/2, -h/2, r, g, b,

                 l/2,  w/2, -h/2, r, g, b,
                 l/2, -w/2, -h/2, r, g, b,
                -l/2, -w/2, -h/2, r, g, b,

                 l/2,  w/2,  h/2, r, g, b,
                -l/2,  w/2,  h/2, r, g, b,
                -l/2, -w/2,  h/2, r, g, b,

                -l/2, -w/2,  h/2, r, g, b,
                 l/2, -w/2,  h/2, r, g, b,
                 l/2,  w/2,  h/2, r, g, b,

                -l/2, -w/2,  h/2, r, g, b,
                -l/2,  w/2,  h/2, r, g, b,
                -l/2,  w/2, -h/2, r, g, b,

                -l/2,  w/2, -h/2, r, g, b,
                -l/2, -w/2, -h/2, r, g, b,
                -l/2, -w/2,  h/2, r, g, b,

                 l/2, -w/2, -h/2, r, g, b,
                 l/2,  w/2, -h/2, r, g, b,
                 l/2,  w/2,  h/2, r, g, b,

                 l/2,  w/2,  h/2, r, g, b,
                 l/2, -w/2,  h/2, r, g, b,
                 l/2, -w/2, -h/2, r, g, b,

                 l/2, -w/2,  h/2, r, g, b,
                -l/2, -w/2,  h/2, r, g, b,
                -l/2, -w/2, -h/2, r, g, b,

                -l/2, -w/2, -h/2, r, g, b,
                 l/2, -w/2, -h/2, r, g, b,
                 l/2, -w/2,  h/2, r, g, b,

                 l/2,  w/2, -h/2, r, g, b,
                -l/2,  w/2, -h/2, r, g, b,
                -l/2,  w/2,  h/2, r, g, b,

                -l/2,  w/2,  h/2, r, g, b,
                 l/2,  w/2,  h/2, r, g, b,
                 l/2,  w/2, -h/2, r, g, b
            )
        self.vertex_count = len(self.vertices)//6
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

    def draw(self, position):
        glUseProgram(self.shader)
        self.material.use()
        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
        model_transform = pyrr.matrix44.multiply(model_transform, pyrr.matrix44.create_from_translation(vec=position,dtype=np.float32))
        glUniformMatrix4fv(glGetUniformLocation(self.shader,"model"),1,GL_FALSE,model_transform)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))

class skyBox:
    def __init__(self, model):
        self.model = model

    def draw(self,position):
        self.model.draw(position)

    def destroy(self):
        self.model.destroy()

app = App()