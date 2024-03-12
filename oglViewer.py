"""
/**         oglViewer.py
 *
 *          Simple Python OpenGL program that uses PyOpenGL + GLFW to get an
 *          OpenGL 3.2 core profile context and animate loaded .obj files.
 *          Everything loads but the Elephant and wasn't sure about a few things here
 *          what does glDrawElements expect? whenever I swapped over I had a weird mesh
 *          so I ordered the Vertices according to the Faces to use glDrawArrays
 *          why does it swap position if I switch shaders? Using the same vertPos calculations...
 *          condition of the model doesn't get saved on purpose, because it bugs out then.
 *          Only the Zoom persists, because I only work with the Y-Axis.
 *          
 ****
"""

import sys
import glfw
import numpy as np

from OpenGL.GL import *
from OpenGL.arrays.vbo import VBO
from OpenGL.GL.shaders import *

from mat4 import *
EXIT_FAILURE = -1
if len(sys.argv) > 1:
    # Loop through the arguments
    FILENAME = sys.argv[1]
    if FILENAME == "elephant.obj":
        print("this won't load, sorry")    
else:
    FILENAME = "squirrel.obj"      


class Scene:
    """
        OpenGL scene class that render a RGB colored tetrahedron.
    """
    
    def read_obj_file(self, filename):
        vertices = []
        normals = []
        self.indices = []
        self.normal_indices = []
        hasNormals = False

        with open(filename, 'r') as obj_file:
            for line in sorted(obj_file.readlines(), key=lambda line: line.split()[0] if line.split() else '', reverse=True):
                if line.startswith('v '):
                    vertex = [float(v) for v in line.split()[1:]]
                    vertices.append(vertex)
                elif line.startswith('vn '):
                    normal = [float(n) for n in line.split()[1:]]
                    hasNormals = True
                    normals.append(normal)
                elif line.startswith('f '):
                    face_data = line.split()[1:]
                    face_indices = []

                    for part in face_data:
                        vertex_index, _, normal_index = part.partition('//')
                        try:
                            vertex_index = int(vertex_index) - 1
                            normal_index = int(normal_index) - 1 if normal_index else None
                            face_indices.append((vertex_index, normal_index))
                        except ValueError:
                            break
                    if len(face_indices) >= 3:
                        v0 = vertices[face_indices[0][0]]
                        v1 = vertices[face_indices[1][0]]
                        v2 = vertices[face_indices[2][0]]
                        face_normal = np.cross(np.subtract(v1, v0), np.subtract(v2, v0))
                        face_normal /= np.linalg.norm(face_normal)
                    for i in range(1, len(face_indices) - 1):
                        self.indices.append(face_indices[0][0])
                        self.indices.append(face_indices[i][0])
                        self.indices.append(face_indices[i + 1][0])
                        if hasNormals:
                            if face_indices[0][1] is not None:
                                self.normal_indices.append(face_indices[0][1])
                                self.normal_indices.append(face_indices[i][1])
                                self.normal_indices.append(face_indices[i + 1][1])
                        else:
                            normals.append(face_normal)
                            normals.append(face_normal)
                            normals.append(face_normal)
                else: 
                    continue
        #ordering the positions and normals according to the self.indices
        self.positions = [vertices[i] for i in self.indices]
        self.normals = [normals[i] for i in self.normal_indices] if hasNormals else normals
        # converting the positions and normals into the needed np.array
        self.positions = np.array(self.positions, dtype=np.float32)
        self.normals = np.array(self.normals, dtype=np.float32)
        # not needed only for gldrawElements
        self.indices = np.array(self.indices, dtype=np.uint32)
        self.normal_indices = np.array(self.normal_indices, dtype=np.uint32)

    
                    

    def __init__(self, width, height, scenetitle="3D Object  Visualizer"):
        self.scenetitle         = scenetitle
        self.width              = width
        self.height             = height
        self.animatable         = True
        self.prev_xpos          = 0.0
        self.prev_ypos          = 0.0
        self.scale              = 1.0
        self.rotation_matrix    = np.mat(4)
        self.translation_matrix = np.mat(4)
        self.axis               = np.array([0.0,1.0,0.0])
        self.angle              = 0
        self.angle_mouse        = 0 
        self.angle_increment    = 1
        self.animate            = False
        self.meshView           = True
        self.phongView          = False
        self.grouraudView        = False
        self.rotate_x           = 0
        self.rotate_y           = 0
        self.rotate_z           = 0
        self.translate_x        = 0
        self.translate_y        = 0
        self.translate_z        = 0
        self.projection_mode    = False
        self.p1x                = 0
        self.p1y                = 0 

    def init_GL(self):
        # setup buffer (vertices, colors, normals, ...)
        self.gen_buffers()

        # setup shader
        glBindVertexArray(self.vao)
        vertex_shader       = open("shader/phongshader.vert","r").read()
        fragment_shader     = open("shader/phongshader.frag","r").read()
        phong_vertex_prog         = compileShader(vertex_shader, GL_VERTEX_SHADER)
        phong_frag_prog           = compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        self.phong_shader_program = compileProgram(phong_vertex_prog, phong_frag_prog)
        vertex_shader       = open("shader/meshshader.vert","r").read()
        fragment_shader     = open("shader/meshshader.frag","r").read()
        mesh_vertex_prog         = compileShader(vertex_shader, GL_VERTEX_SHADER)
        mesh_frag_prog           = compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        self.mesh_shader_program = compileProgram(mesh_vertex_prog, mesh_frag_prog)
        vertex_shader       = open("shader/grouraudshader.vert","r").read()
        fragment_shader     = open("shader/grouraudshader.frag","r").read()
        grouraud_vertex_prog         = compileShader(vertex_shader, GL_VERTEX_SHADER)
        grouraud_frag_prog           = compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        self.grouraud_shader_program = compileProgram(grouraud_vertex_prog, grouraud_frag_prog)

        # unbind vertex array to bind it again in method draw
        glBindVertexArray(0)
    
    def gen_buffers(self):
        # TODO: 
        # 1. Load geometry from file and calc normals if not available
        # 2. Load geometry and normals in buffer objects
        
        self.read_obj_file("models/" + FILENAME)
        # VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # VBO for positions
        self.positions_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.positions_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.positions.nbytes, self.positions, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # VBO for normals
        self.normals_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.normals_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.normals.nbytes, self.normals, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # EBO (if needed)
        #self.ebo = glGenBuffers(1)
        #glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        #glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_READ)

        # Unbind buffers and VAO
        #glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)



    def set_size(self, width, height):
        self.width = width
        self.height = height


    def draw(self):
        glEnable(GL_DEPTH_TEST)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # Get the bounding box of the object
        min_x, max_x = np.min(self.positions[:, 0]), np.max(self.positions[:, 0])
        min_y, max_y = np.min(self.positions[:, 1]), np.max(self.positions[:, 1])
        min_z, max_z = np.min(self.positions[:, 2]), np.max(self.positions[:, 2])

        # Calculate the center and size of the object
        center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2])
        self.size = max(max_x - min_x, max_y - min_y, max_z - min_z)

        # Calculate the distance from the object based on its size
        distance = self.size * 2 + self.size * 0.5
        # Calculate the new view matrix based on the object's center and distance
        view = look_at(center[0], center[1], center[2] + distance, center[0], center[1], center[2], 0, 1, 0)
        
    
        self.model = (
            translate(center[0], center[1], center[2] - distance)
            @scale(self.scale, self.scale, self.scale)
            @ rotate_x(self.rotate_x)
            @ rotate_y(self.rotate_y)
            @ rotate_z(self.rotate_z)
            @ rotate(self.angle_mouse, self.axis)
            @ translate(self.translate_x,self.translate_y,self.translate_z)
        )
        if self.projection_mode:
            # Calculate the aspect ratio based on the current width and height
            aspect_ratio = self.width / self.height
            
            # Calculate the new values for left, right, bottom, and top
            left = -2 * aspect_ratio
            right = 2 * aspect_ratio
            bottom = -2
            top = 2
            # Update the projection matrix
            projection = ortho(left, right, bottom, top, 0.1, 100.0)
        else:
            # Calculate the aspect ratio based on the current width and height
            aspect_ratio = self.width / self.height
            
            # Update the projection matrix
            projection = perspective(45.0, aspect_ratio, 1.0, 100.0)

        
        mvp_matrix = projection @ view @ self.model
        normal_matrix = np.transpose(np.linalg.inv(self.model))
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.positions_vbo)
        #glBindBuffer(GL_ARRAY_BUFFER, self.normals_vbo)
        #Needed for glDrawElements but i couldn't get it working properly
        #glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        if self.animate:
            self.angle += self.angle_increment
        if self.meshView:
            glUseProgram(self.mesh_shader_program)
            varLocation = glGetUniformLocation(self.mesh_shader_program, 'modelview_projection_matrix')
            glUniformMatrix4fv(varLocation, 1, GL_TRUE, mvp_matrix)
            varLocation = glGetUniformLocation(self.mesh_shader_program, 'model_matrix')
            glUniformMatrix4fv(varLocation, 1, GL_TRUE, self.model)
            varLocation = glGetUniformLocation(self.mesh_shader_program, 'normal_matrix')
            glUniformMatrix3fv(varLocation, 1, GL_TRUE, normal_matrix)

            # (a) Render wireframe model
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glDrawArrays(GL_TRIANGLES, 0, len(self.indices))
            #glDrawElements(GL_TRIANGLES, len(self.combined_indices), GL_UNSIGNED_INT, None)
        if self.grouraudView:
            glUseProgram(self.grouraud_shader_program)
            # enable shader & set uniforms
            varLocation = glGetUniformLocation(self.grouraud_shader_program, 'modelview_projection_matrix')
            glUniformMatrix4fv(varLocation, 1, GL_TRUE, mvp_matrix)

            varLocation = glGetUniformLocation(self.grouraud_shader_program, 'modelview_matrix')
            glUniformMatrix4fv(varLocation, 1, GL_TRUE, self.model)

            varLocation = glGetUniformLocation(self.grouraud_shader_program, 'normal_matrix')
            glUniformMatrix3fv(varLocation, 1, GL_TRUE, normal_matrix)

            varLocation = glGetUniformLocation(self.grouraud_shader_program, 'shininess')
            glUniform1f(varLocation, 15.0)

            varLocation = glGetUniformLocation(self.grouraud_shader_program, 'ambientColor')
            glUniform3f(varLocation,0.49, 0.44, 0.16)

            varLocation = glGetUniformLocation(self.grouraud_shader_program, 'diffuseColor')
            glUniform3f(varLocation, 0.62, 0.56, 0.21)

            varLocation = glGetUniformLocation(self.grouraud_shader_program, 'specularColor')
            glUniform3f(varLocation, 1.0, 1.0, 1.0)

            varLocation = glGetUniformLocation(self.grouraud_shader_program, 'light_position')
            glUniform3f(varLocation, 1.0, 1.0, 1.0)

            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glDrawArrays(GL_TRIANGLES, 0, len(self.indices))
        if self.phongView:
            # enable shader & set uniforms
            glUseProgram(self.phong_shader_program)

            varLocation = glGetUniformLocation(self.phong_shader_program, 'modelview_projection_matrix')
            glUniformMatrix4fv(varLocation, 1, GL_TRUE, mvp_matrix)
            varLocation = glGetUniformLocation(self.phong_shader_program, 'model_matrix')
            glUniformMatrix4fv(varLocation, 1, GL_TRUE, self.model)
            varLocation = glGetUniformLocation(self.phong_shader_program, 'normal_matrix')
            glUniformMatrix3fv(varLocation, 1, GL_TRUE, normal_matrix)
            varLocation = glGetUniformLocation(self.phong_shader_program, 'shininess')
            glUniform1f(varLocation, 15.0)
            varLocation = glGetUniformLocation(self.phong_shader_program, 'ambient_color')
            glUniform3f(varLocation, 0.49, 0.44, 0.16) # darker
            varLocation = glGetUniformLocation(self.phong_shader_program, 'diffuse_color')
            glUniform3f(varLocation, 0.62, 0.56, 0.21)
            varLocation = glGetUniformLocation(self.phong_shader_program, 'light_color')
            glUniform3f(varLocation, 1.0, 1.0, 1.0)  # Example light color (white)
            varLocation = glGetUniformLocation(self.phong_shader_program, 'light_position')
            glUniform3f(varLocation, .5, 1.0, 1.0)  # Example light position

            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glDrawArrays(GL_TRIANGLES, 0, len(self.indices))
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        # unbind the shader
        glUseProgram(0)

    def rotate_x_axis(self):
        self.rotate_x += 10

    def rotate_y_axis(self):
        self.rotate_y += 10

    def rotate_z_axis(self):
        self.rotate_z += 10
    
    # Helpfunction to projekt on a Sphere
    def projectOnSphere(self, x, y):
        r = min(self.width, self.height) * 0.4
        x, y = x - self.width/2.0, self.height/2.0 - y
        a = min(r*r, x**2 + y**2)
        z = np.sqrt(r*r - a)
        l = np.sqrt(x**2 + y**2 + z**2)
        return x/l, y/l, z/l
    
    def rotate_on_Drag(self, p1x, p1y, p2x, p2y):
        if p2x != 0 and p2y != 0:
            projected_p1x, projected_p1y, p1z = self.projectOnSphere(p1x, p1y)
            projected_p2x, projected_p2y, p2z = self.projectOnSphere(p2x, p2y)
            dx = p2x - self.p1x
            dy = p2y - self.p1y
            self.p1x = p2x
            self.p1y = p2y
            self.axis = np.cross([projected_p1x, projected_p1y, p1z], [projected_p2x, projected_p2y, p2z])

            # Calculate the dot product
            dot_product = np.dot([projected_p1x, projected_p1y, p1z], [projected_p2x, projected_p2y, p2z])

            # Check for invalid values in the dot product
            dot_product = np.clip(dot_product, -1.0, 1.0)

            # Calculate the angle using arccos
            self.angle_mouse = np.arccos(dot_product) * 100.0  # Adjust the scaling factor as needed

            # Apply the rotation to a temporary matrix
            self.rotation_matrix = rotate(self.angle_mouse, self.axis)

            # Apply the temporary rotation matrix to the main model matrix
            self.model = self.rotation_matrix @ self.model
    
    def zoom_on_Drag(self, p2y):
        if p2y != 0:
            dy = p2y - self.p1y
            self.p1y = p2y
            distance = np.sqrt(dy**2)           
            # Set a zoom sensitivity factor (adjust as needed)
            zoom_sensitivity = 0.01
            # Calculate the zoom factor based on the distance and sensitivity
            zoom_factor = 1.0 + distance * zoom_sensitivity

            # Adjust the zoom factor to handle zooming in and out
            if dy < 0:
                zoom_factor = 1.0 / zoom_factor

            self.scale *= zoom_factor
            
            self.model = scale(self.scale, self.scale, self.scale) @ self.model
    
    def parallel_on_Drag(self, p1x, p1y, p2x, p2y):
        if p2x != 0 and p2y != 0:
            dx = p2x - p1x
            dy = p2y - p1y
            dy *= -1
            self.angle_mouse = 0
            # Set a translation sensitivity factor (adjust as needed)
            translation_sensitivity = 0.005
            
            # Calculate the translation amount based on the mouse movement and sensitivity
            self.translate_x = dx * translation_sensitivity
            self.translate_y = dy * translation_sensitivity
            self.translate_z = 0.0  # Set the z-translation to 0 (parallel to the image plane)
            
            # Create the translation matrix using the translate function
            self.translation_matrix = translate(self.translate_x, self.translate_y, self.translate_z) 
            
            # Apply the translation matrix to the main model matrix
            self.model = self.translation_matrix @ self.model
               


class RenderWindow:
    """
        GLFW Rendering window class
    """

    def __init__(self, scene):
        # initialize GLFW
        if not glfw.init():
            sys.exit(EXIT_FAILURE)

        # request window with old OpenGL 3.2
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 2)

        # make a window
        self.width, self.height = scene.width, scene.height
        self.aspect = self.width / self.height
        self.window = glfw.create_window(self.width, self.height, scene.scenetitle, None, None)
        
        if not self.window:
            glfw.terminate()
            sys.exit(EXIT_FAILURE)

        # Make the window's context current
        glfw.make_context_current(self.window)

        # initialize GL
        self.init_GL()

        # set window callbacks
        glfw.set_mouse_button_callback(self.window, self.on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self.on_cursor_move)
        glfw.set_key_callback(self.window, self.on_keyboard)
        glfw.set_window_size_callback(self.window, self.on_size)

        # create scene
        self.scene = scene  
        if not self.scene:
            glfw.terminate()
            sys.exit(EXIT_FAILURE)

        self.scene.init_GL()

        # exit flag
        self.exitNow = False
        self.left_button_pressed = False
        self.right_button_pressed = False
        self.middle_button_pressed = False
        self.p1x, self.p1y = (0,0)


    def init_GL(self):
        # debug: print GL and GLS version
        # print('Vendor       : %s' % glGetString(GL_VENDOR))
        # print('OpenGL Vers. : %s' % glGetString(GL_VERSION))
        # print('GLSL Vers.   : %s' % glGetString(GL_SHADING_LANGUAGE_VERSION))
        # print('Renderer     : %s' % glGetString(GL_RENDERER))

        # set background color to white
        glClearColor(1.0, 1.0, 1.0, 1.0)     

        # Enable depthtest
        glEnable(GL_DEPTH_TEST)


    def on_mouse_button(self, win, button, action, mods):
        
        if action == glfw.PRESS:
            if button == glfw.MOUSE_BUTTON_LEFT: 
                self.p1x, self.p1y = glfw.get_cursor_pos(win)
                scene.p1x, scene.p1y = self.p1x, self.p1y
                self.left_button_pressed = True
            if button == glfw.MOUSE_BUTTON_RIGHT: 
                self.p1x, self.p1y = glfw.get_cursor_pos(win)
                self.right_button_pressed = True
            if button == glfw.MOUSE_BUTTON_MIDDLE: 
                print("Hold the Button and pull down to scroll in and pull up to scroll out")
                self.p1x, self.p1y = glfw.get_cursor_pos(win)
                scene.p1x, scene.p1y = self.p1x, self.p1y
                self.middle_button_pressed = True
        elif action == glfw.RELEASE:
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.left_button_pressed = False
            if button == glfw.MOUSE_BUTTON_RIGHT: 
                self.right_button_pressed = False
            if button == glfw.MOUSE_BUTTON_MIDDLE: 
                self.middle_button_pressed = False
        
    
    def on_cursor_move(self, win, button, action):
        xCoord, yCoord = glfw.get_cursor_pos(win)
        if xCoord < scene.width and xCoord > 0 and yCoord <scene.height and yCoord > 0:
            p2x, p2y = glfw.get_cursor_pos(win)
        else:
            p2x, p2y = (0,0)
            
        if self.left_button_pressed:
            scene.rotate_on_Drag(self.p1x, self.p1y, p2x, p2y)
        if self.middle_button_pressed:
            scene.zoom_on_Drag(p2y)
        if self.right_button_pressed:
            scene.parallel_on_Drag(self.p1x, self.p1y, p2x, p2y)
        
        
            
    def on_keyboard(self, win, key, scancode, action, mods):
        print("keyboard: ", win, key, scancode, action, mods)
        rotation_speed = 2.0 
        if action == glfw.PRESS:
            # ESC to quit
            if key == glfw.KEY_ESCAPE:
                self.exitNow = True
            if key == glfw.KEY_A:
                self.scene.animate = not self.scene.animate
            if key == glfw.KEY_P:
                self.scene.projection_mode = not self.scene.projection_mode
                print("toggle projection: orthographic / perspective ")
            if key == glfw.KEY_S:
                if self.scene.phongView:
                    self.scene.phongView = not self.scene.phongView
                    self.scene.meshView = not self.scene.meshView
                elif self.scene.grouraudView:
                    self.scene.grouraudView = not self.scene.grouraudView
                    self.scene.phongView = not self.scene.phongView
                elif self.scene.meshView:
                    self.scene.meshView = not self.scene.meshView
                    self.scene.grouraudView = not self.scene.grouraudView
                print("toggle shading: wireframe, grouraud, phong")
            if key == glfw.KEY_X and action == glfw.PRESS:
                scene.rotate_x_axis()
                print("rotate: around x-axis")
            if key == glfw.KEY_Y and action == glfw.PRESS:
                scene.rotate_y_axis()
                print("rotate: around y-axis")
            if key == glfw.KEY_Z and action == glfw.PRESS:
                scene.rotate_z_axis()
                print("rotate: around z-axis")


    def on_size(self, win, width, height):
        self.scene.set_size(width, height)


    def run(self):
        while not glfw.window_should_close(self.window) and not self.exitNow:
            # poll for and process events
            glfw.poll_events()

            # setup viewport
            width, height = glfw.get_framebuffer_size(self.window)
            glViewport(0, 0, width, height);

            # call the rendering function
            self.scene.draw()
            
            # swap front and back buffer
            glfw.swap_buffers(self.window)

        # end
        glfw.terminate()



# main function
if __name__ == '__main__':

    print("presse 'a' to toggle animation...")

    # set size of render viewport
    width, height = 640, 480

    # instantiate a scene
    scene = Scene(width, height)

    # pass the scene to a render window ... 
    rw = RenderWindow(scene)

    # ... and start main loop
    rw.run()
