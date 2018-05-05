import numpy as np
import subprocess
import sys
import openmesh
import os
from scipy.spatial.distance import cdist


def pair_distances(backbone, groupB):
    backbone_points = np.zeros((len(backbone), 3))
    for i in range(len(backbone)):
        backbone_points[i] = backbone[i].point
    return cdist(backbone_points, groupB, 'euclidean')

def unit_normal(a, b, c):
    x = np.linalg.det([[1,a[1],a[2]],
         [1,b[1],b[2]],
         [1,c[1],c[2]]])
    y = np.linalg.det([[a[0],1,a[2]],
         [b[0],1,b[2]],
         [c[0],1,c[2]]])
    z = np.linalg.det([[a[0],a[1],1],
         [b[0],b[1],1],
         [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    return (x/magnitude, y/magnitude, z/magnitude)

#area of polygon poly
def poly_area(poly):
    if len(poly) < 3: # not a plane - no area
        return 0
    total = [0, 0, 0]
    N = len(poly)
    for i in range(N):
        vi1 = poly[i]
        vi2 = poly[(i+1) % N]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)


def get_path(src_coor, src_face, target_coor, target_face, filename):
    #Returns path as list of arrays of 3D points
    args = ("./cgal/shortest_paths", filename, 
        str(src_face), str(src_coor[0]), str(src_coor[1]), str(src_coor[2]),
        str(target_face), str(target_coor[0]), str(target_coor[1]), str(target_coor[2]))
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    with open ("shortest_path.txt", "r") as myfile:
        points=myfile.readlines()

    for i in range(len(points)):
        xyz = points[i].split()
        for j in range(len(xyz)):
            xyz[j] = float(xyz[j])
        points[i] = np.array(xyz)
    return points


class Node:
    NUM_CALLS = 0

    def __init__(self, point, bary_coor, face):
        self.parent = None
        self.point = point
        self.bary_coor = bary_coor
        self.face = face
        self.children = []

    def add_child(self, obj):
        obj.parent = self
        self.children.append(obj)

    def render_tree(self, filename, path_arrs, level = 0):

        Node.NUM_CALLS += 1
        if Node.NUM_CALLS % 50 == 0:
            print("Rendered {} samples".format(Node.NUM_CALLS))

        if level == 0:
            path_arrs[0] = [np.array([level, level, level])]

        cntr = 0
        for child in self.children:
            a_coor = self.bary_coor
            a_face = self.face
            b_coor = child.bary_coor
            b_face = child.face
            points = get_path(a_coor, a_face, b_coor, b_face, filename);
            rev_points = points[::-1]

            if cntr == 0:
                if len(child.children) > 0:
                    path_arrs[-1].extend(rev_points[:-1])
                else:
                    path_arrs[-1].extend(rev_points)
            else:
                if len(child.children) > 0:
                    new_path = rev_points[:-1]
                    new_path = [np.array([level, level, level])] + new_path
                    path_arrs.append(new_path)
                else:
                    new_path = rev_points
                    new_path = [np.array([level, level, level])] + new_path
                    path_arrs.append(new_path)
            

            child.render_tree(filename, path_arrs, level + (len(points) - 1))
            cntr += 1 
    
class Growth:
    def __init__(self, filename, num_samples, sort_axis=2):
        print("reading mesh")
        self.read_mesh(filename+".off")
        self.num_samples = num_samples
        self.filename = filename+".off"
        print("calc edges")
        self._init_edges()
        print("calc areas")
        self.struc_axis = sort_axis
        self._init_areas()
        self.path = [[]]
        self.weigh_bottom = False

    def read_mesh(self, filename):
        args = ("./cgal/read_mesh", filename)
        popen = subprocess.Popen(args, stdout=subprocess.PIPE)
        popen.wait()
        with open ("mesh.txt", "r") as myfile:
            values=myfile.readlines()
        print("manual conversions")
        if (len(values) == 2):
            print("Mesh Broken, fixing with openscad")
        num_vertices = 0
        vertices = 0
        num_faces = 0
        faces_arr = []
        for i in range(len(values)):
            if i == 0:
                num_vertices = int(values[i])
                vertices = np.zeros((num_vertices, 3))
                continue
            if i <= num_vertices:
                vertex = []
                for coor in values[i].split():
                    vertex.append(float(coor))
                vertices[i-1] = np.array(vertex)
                continue
            if i == num_vertices + 1:
                num_faces = int(values[i])
                continue
            face_vert = values[i][1:]
            if face_vert:
                faces_arr.append(int(face_vert))
        faces = np.array([np.array(faces_arr[n:n+3]) for n in range(0, len(faces_arr), 3)])
        self.points = vertices
        self.faces = faces
        
    def _init_edges(self):
        faces = self.faces
        edges = np.zeros((faces.shape[0] * 3, 2))
        i = 0
        for face in faces:
            edges[i] = np.array([min(face[0], face[1]),
                                max(face[0], face[1])])
            i +=1
            edges[i] = np.array([min(face[0], face[2]),
                                max(face[0], face[2])])
            i +=1
            edges[i] = np.array([min(face[1], face[2]),
                                max(face[1], face[2])])
            i +=1
        self.edges = np.unique(edges, axis=0)

    def _init_areas(self):
        faces = self.faces
        areas = np.zeros(len(faces))
        strucs = np.zeros(len(faces))
        for j in range(len(faces)):
            face = faces[j]
            verts = [self.points[i] for i in face]
            strucs[j] = min(verts, key=lambda v:v[self.struc_axis])[self.struc_axis]
            area = poly_area(verts)
            areas[j] = area
        self.areas = areas
        strucs -= np.amax(strucs)
        strucs *= -1
        self.strucs = strucs


    def sample_face(self, face_index):
        r1 = np.random.random()
        r2 = np.random.random()
        face = [self.points[i] for i in self.faces[face_index]]
        if r1 + r2 >=1:
            r1 = 1 - r1
            r2 = 1 - r2
        point = face[0] + r1 * (face[1] - face[0]) + r2 * (face[2] - face[0])
        return (point, np.array([r2, 1-r1-r2, r1]))

    def sample_mesh(self, num_samples):
      # compute and sum triangle areas
      samples = np.zeros((num_samples,3))
      bary_coors = np.zeros((num_samples,3))
      areas = self.areas
      prob_area = np.ones(areas.shape) / areas.shape[0]
      prob_area = areas / np.sum(areas)
      if self.weigh_bottom:
        prob_area = self.strucs / sum(self.strucs)
      else:
        prob_area = areas / np.sum(areas)
      selected_faces = np.random.choice(np.arange(len(areas)), num_samples, p=prob_area)
      for i in range(num_samples):
        samples[i], bary_coors[i] = self.sample_face(selected_faces[i])
      return samples, bary_coors, selected_faces


    def write_mesh_to_file(self):
        vertices = self.points
        faces = self.faces
        num_vertices = len(vertices)
        num_faces = len(faces)
        print(num_faces)
        print(num_vertices)
        print(vertices)
        print(faces)
        
    def run_alg(self):
        num_samples = self.num_samples
        self.samples, self.bary_coors, self.sample_faces = self.sample_mesh(num_samples)

        tree_samples = self.samples.copy()
        tree_bary = self.bary_coors.copy()
        tree_face = self.sample_faces.copy()

        src_index = 0
        src = Node(tree_samples[src_index], tree_bary[src_index], tree_face[src_index])

        tree_samples = np.delete(tree_samples, src_index, 0)
        tree_bary = np.delete(tree_bary, src_index, 0)
        tree_face = np.delete(tree_face, src_index, 0)

        self.backbone = [src]        

        while len(tree_samples) > 0:
            dist = pair_distances(self.backbone, tree_samples)
            ind = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
            to_del = ind[1]
            new_node = Node(tree_samples[to_del], tree_bary[to_del], tree_face[to_del])
            back_node = self.backbone[ind[0]]
            back_node.add_child(new_node)
            self.backbone.append(new_node)
            tree_samples = np.delete(tree_samples, to_del, 0)
            tree_bary = np.delete(tree_bary, to_del, 0)
            tree_face = np.delete(tree_face, to_del, 0)

        self.tree = src
    
    def render_path(self):

        # Render Path tree
        self.tree.render_tree(self.filename, self.path)
        for i in range(len(self.path)):
            self.path[i] = np.array(self.path[i])
            # print(self.path[i])
            # print("\n")

        short_name = self.filename.split("/")[-1][:-4]
        directory = "growth_paths"
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            for path_file in os.listdir(directory):
                file_path = os.path.join(directory, path_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)

        
        for i, path in enumerate(self.path):
            # Output to text files for the re-rendering
            np.savetxt("%s/path%s.txt" % (directory, i), path)

def render_stl():
    args = ("./run_blender.sh")
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read().decode("utf-8")
    lines = output.splitlines()
    for line in lines:
        print(line)

def convert_to_off(filename):
    """ Converts file from stl to off"""
    mesh = openmesh.read_trimesh(filename+".stl")
    openmesh.write_mesh(filename+".off", mesh)


def main(argv):
    if len(argv) == 0:
        filename = "input/sphere"
    else:
        filename = argv[0]
        if not filename or filename[-4:] != ".stl":
            print(filename+" is not a valid STL file")
            return
        filename = filename[:-4]

    if len(argv) < 2:
        rand_seed = 10
    else:
        rand_seed = int(argv[1])
    np.random.seed(rand_seed)

    if len(argv) < 3:
        num_samples = 200
    else:
        num_samples = int(argv[2])

    with open("title.txt", "w") as text_file:
        short_name = filename.split("/")[-1]
        text_file.write(short_name+"_"+str(rand_seed)+"_"+str(num_samples))

    convert_to_off(filename)

    print("reading in file")
    g = Growth(filename, num_samples)

    print("start alg")
    g.run_alg()
    print("start render to "+ str(num_samples)+" samples ...")
    g.render_path()
    print("Exporting to stl ...")
    render_stl()
    os.remove(filename+".off")


if __name__ == '__main__':
    main(sys.argv[1:])

    # Example call: python grow_hyphae.py input/sphere.stl 10 500
