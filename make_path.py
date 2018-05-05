#!/usr/bin/env python3

import bpy
import numpy as np
import os

MIN_RAD = 0.4
DEC_PERC = 0.995
START_RAD = 0.5

class Path(object):
  """Creates a 3D path from input points"""
  def __init__(self, points, title):
    super(Path, self).__init__()
    self.level = int(points[0][0])
    self.points = points[1:]
    self.title = title
    dec_rad = START_RAD * (DEC_PERC ** self.level)
    self.radius = max(MIN_RAD, dec_rad)
    self.obj = None


  def render_curve(self):

    # data
    coords = self.points

    # create the Curve Datablock
    curveData = bpy.data.curves.new('treeCurve', type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = 2
    curveData.use_fill_caps = True

    #Add Bevel object
    tor0refCirc = vecCircle("tor0refCirc", self.radius, 'CURVE')
    curveData.bevel_object = tor0refCirc

    #Add Taper Object
    dec_perc = DEC_PERC ** len(self.points)
    if (dec_perc * self.radius <= MIN_RAD):
      dec_perc = MIN_RAD / self.radius
    taper_obj = taperCurve(dec_perc)
    curveData.taper_object = taper_obj


    # map coords to spline
    polyline = curveData.splines.new('NURBS')
    polyline.use_endpoint_u = True
    polyline.points.add(len(coords)-1)
    for i, coord in enumerate(coords):
        x,y,z = coord
        polyline.points[i].co = (x, y, z, 1)

    # create Object
    curveOB = bpy.data.objects.new(self.title, curveData)


    # attach to scene and validate context
    scn = bpy.context.scene
    scn.objects.link(curveOB)
    scn.objects.active = curveOB
    curveOB.select = True
    self.obj = curveOB


def delete_default():
  '''
  Removes default objects from the scene.
  '''

  names = ['Cube', 'Icosphere', 'Sphere']
  for name in names:
    try:
      bpy.data.objects[name].select = True
      bpy.ops.object.delete()
    except Exception:
      pass

  return

def save_blend(fn):

  bpy.ops.wm.save_as_mainfile(filepath=fn)

  return

def taperCurve(dec_perc):

  bpy.ops.curve.primitive_nurbs_path_add(location=(0.0, 0.0, 0.0))
  obj = bpy.context.active_object
  obj.name = "taperCurve"
  num_points = len(obj.data.splines[0].points)
  inc = (1.0 - dec_perc) / num_points
  for i in range(num_points):
    co = obj.data.splines[0].points[i].co
    obj.data.splines[0].points[i].co = (co[0], 1.0 - i*inc, 0.0, 1.0)

  return obj


def vecCircle(name, radius, obj_type='MESH'):
  num_verts = 8
  bpy.ops.mesh.primitive_circle_add(vertices=num_verts, radius=radius)
  obj = bpy.context.active_object
  obj.name = name
  if obj_type == 'CURVE':
      bpy.ops.object.convert(target='CURVE', keep_original=False)
  return obj


def export_stl(curves):

  # get the current path and make a new folder for the exported meshes
  path = bpy.path.abspath('//stlexport/')

  if not os.path.exists(path):
      os.makedirs(path)

  # deselect all meshes
  bpy.ops.object.select_all(action='DESELECT')

  for curve in curves:
    if curve.obj:
      # select the object
      curve.obj.select = True

  print("exporting to stl ...")
  # export object with its name as file name
  fPath = str((path + curves[0].title + '.stl'))
  bpy.ops.export_mesh.stl(filepath=fPath)

def make_single_path(filename):
  points = np.loadtxt(filename)
  if points.shape == (3,):
      print("Hole found in mesh")
      return None

  with open("title.txt") as f:
    content = f.readlines()
    title = content[0] + "_growth"

  P = Path(points, title)
  P.render_curve()
  return P

def join_all(curves):
  # deselect all meshes
  bpy.ops.object.select_all(action='DESELECT')

  for curve in curves:
    if curve.obj:
      # select the object
      curve.obj.select = True

  #Convert to mesh
  bpy.ops.object.convert(target='MESH')

  #join all curves
  bpy.ops.object.join()

  return bpy.context.selected_objects[0]

def animate_all(curves):
  #set duration and speed
  frame_chunk = 0.5
  max_level = max([(curve.level + len(curve.points)) for curve in curves])
  duration = max_level * frame_chunk

  # prepare a scene
  scn = bpy.context.scene
  scn.frame_start = 1
  scn.frame_end = duration
  

  for curve in curves:
    animate(curve, frame_chunk)


def animate(curve, frame_chunk):
  # deselect all objects
  bpy.ops.object.select_all(action='DESELECT')
  
  # select the object
  curve.obj.select = True

  start_frame = 1 + (curve.level*frame_chunk)

  num_steps = len(curve.points)
  bevel_chunk = 1.0 / num_steps

  if (num_steps >= 3):
    iter_list = [0, int(num_steps/2.0), int(num_steps) - 1]
  else:
    iter_list = range(int(num_steps))

  for i in iter_list:
    # move to frame
    cur_fram = (i * frame_chunk) + start_frame
    bpy.context.scene.frame_set(cur_fram)
    curve.obj.data.bevel_factor_end = i * bevel_chunk
    curve.obj.data.keyframe_insert(data_path="bevel_factor_end",frame=cur_fram)


def assign_materials(curves):
  # Get material
  mat = bpy.data.materials.get("Material")
  if mat is None:
      # create material
      mat = bpy.data.materials.new(name="Material")

  for curve in curves:
    ob = curve.obj
    # Assign it to object
    if ob.data.materials:
        # assign to 1st material slot
        ob.data.materials[0] = mat
    else:
        # no slots
        ob.data.materials.append(mat)
    ob.active_material.diffuse_color = (0.093986, 0.386677, 0.8)
    # ob.active_material.diffuse_color = (0.00810887, 0.661868, 0.0641863)


def main():
  from time import time

  t1 = time()
  delete_default()

  paths = []
  for file in os.listdir("./growth_paths"):
    if file.endswith(".txt"):
        filename = os.path.join("./growth_paths", file)
        path = make_single_path(filename)
        if path != None:
          paths.append(path)

  print('\nAlg Time:',time()-t1,'\n\n')
  #export File as stl
  export_stl(paths)

  # Add animation keyframes to belnder file
  animate_all(paths)
  assign_materials(paths)

  #save blend file
  blend_name = "./blend/" + paths[0].title + ".blend"
  save_blend(blend_name)

  print('\nTotal time:',time()-t1,'\n\n')


  return 0;


if __name__ == '__main__':
  main()




