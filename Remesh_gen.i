[Mesh]
  # Load the mesh
  [input_mesh]
    type = FileMeshGenerator
    file = vac_hive.e
    #refine = 1
  []

  # Add sidesets from normals
  [sidesets_from_normals]
    type = SideSetsFromNormalsGenerator
    input = input_mesh
    included_subdomains = 'vacuum_region'
    new_boundary = 'exterior1 exterior2 exterior3 exterior4 exterior5 terminal_plane'
    normals = '0 0 -1   1 1 0   -1 1 0   -1 -1 0   1 -1 0  0 0 1'
    output=true
    show_info = true
    normal_tol = 0.1

  []

[]

[Executioner]
  type = Steady
[]

[Outputs]
  file_base = New_Mesh_Exterior
  exodus = true
[]



