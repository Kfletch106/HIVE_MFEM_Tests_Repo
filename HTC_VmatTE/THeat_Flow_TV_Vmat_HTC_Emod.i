!include Parameters.i

[Mesh]
  type = FileMesh
  file = ../vac_hive.e
  #uniform_refine =1
[]

[Variables]
  [T]
    family = LAGRANGE
    order = FIRST
    initial_condition = ${t_initial}
  []
[]

[Problem]
  verbose_multiapps = true
[]

[AuxVariables]
  [P]
      family = MONOMIAL
      order = CONSTANT
  []

  [wall_heat_flux]
      family = MONOMIAL
      order = CONSTANT
      initial_condition = 0
  []

  [T_avg]
    family = MONOMIAL
    order = CONSTANT
    initial_condition = ${room_temperature}
  []
[]

[Kernels]
  [HeatConduction]
    type = HeatConduction
    variable = T
  []
  # [TimeDerivative]
  #   type = HeatConductionTimeDerivative
  #   variable = T
  # []
  [HeatSource]
    type = CoupledForce
    variable = T
    v = P
    block = monoblock
  []
[]

[Materials]
  # Thermal conductivity function
  [k]
    type = ParsedMaterial
    expression = 'k0 + a*(T - 300)'
    constant_names = 'k0 a'
    constant_expressions = '15.0 0.01'
    coupled_variables = 'T'
    block = 'coil monoblock'
    property_name = thermal_conductivity
  []

  # Specific heat function
  [c_func]
    type = ParsedMaterial
    expression = 'c0 + b*T'
    constant_names = 'c0 b'
    constant_expressions = '500.0 0.2'
    coupled_variables = 'T'
    block = 'coil monoblock'
    property_name = specific_heat
  []

  # Density function
  [rho_func]
    type = ParsedMaterial
    expression = 'rho0*(1 - beta*(T - 300))'
    constant_names = 'rho0 beta'
    constant_expressions = '7800 1e-4'
    coupled_variables = 'T'
    block = 'coil monoblock'
    property_name = density
  []
[]

[Materials]
  #  [steel]
  #   type = GenericFunctionMaterial
  #   prop_names =  'thermal_conductivity   specific_heat     density'
  #   prop_values = 'k_func    c_func      rho_func'
  #   block = 'coil monoblock'
  # []
  [vacuum]
    type = GenericConstantMaterial
    prop_names =  'thermal_conductivity    specific_heat      density'
    prop_values = '${vacuum_tconductivity} ${vacuum_capacity} ${vacuum_density}'
    block = vacuum_region
  []
[]

[BCs]
  # [plane]
  #   type = DirichletBC
  #   variable = T
  #   boundary = 'voltage-surf-1 voltage-surf-2 terminal_plane'
  #   value = ${room_temperature}
  # []
  # Use the fluid wall temperature as a matched value boundary condition.
  [fluid_interface]
    type = ConvectiveHeatFluxBC
    variable = T
    boundary = monoblock_htc #pipe_inner
    T_infinity = ${Fluid_Temp}
    heat_transfer_coefficient = ${HTC_block}
  []
[]

[Postprocessors]
  [P(total){W}]
    type = ElementIntegralVariablePostprocessor
    variable = P
    block = monoblock
  []
  [P(Max){W.m-3}]
    type = ElementExtremeValue
    variable = P
    block = monoblock
  []
  [T_avg]
    type = ElementAverageValue
    variable = T
    block = monoblock
    execute_on = 'initial timestep_begin timestep_end'
  []
  [T(Max){K}]
    type = NodalExtremeValue
    variable = T
    block = monoblock
  []
  [Q_surf]
    type = SideDiffusiveFluxIntegral
    variable = T
    boundary = monoblock_htc #pipe_inner
    diffusivity = thermal_conductivity
  []
  [ThermoC_1]
    type = PointValue
    variable = T
    point = '0.0116 -0.0245 0.0194'
  []
  [ThermoC_2]
    type = PointValue
    variable = T
    point = '0.0138 -0.0245 0.0013'
  []
  [ThermoC_3]
    type = PointValue
    variable = T
    point = '0.0067 -0.0245 0.0124'
  []
  [ThermoC_4]
    type = PointValue
    variable = T
    point = '0.0110 0.0245 0.0031'
  []
  [ThermoC_5]
    type = PointValue
    variable = T
    point = '-0.0105 0.0245 -0.005'
  []
  [ThermoC_6]
    type = PointValue
    variable = T
    point = '-0.0058 0.0245 0.0171'
  []
  [ThermoC_7]
    type = PointValue
    variable = T
    point = '-0.018 -0.0006 0.0164'
  []
  [ThermoC_8]
    type = PointValue
    variable = T
    point = '-0.018 -0.004 -0.0085'
  []
  [ThermoC_9]
    type = PointValue
    variable = T
    point = '-0.018 -0.0047 0.0073'
  []
  [ThermoC_10]
    type = PointValue
    variable = T
    point = '-0.018 0.0124 -0.0032'
  []
[]

[Executioner]
  type = Transient
  solve_type = NEWTON
  petsc_options_iname = '-pc_type -ksp_rtol'
  petsc_options_value = 'hypre    1e-12'
  start_time = 0.0
  nl_abs_tol=1e-8
  l_abs_tol = 1e-8
  end_time = ${end_t_th}
  dt = ${delta_t_th}
  steady_state_detection = true
  steady_state_tolerance = 1e-6
  line_search = none
  automatic_scaling=true
[]

[Outputs]
  csv    = true

  [ex]
    type = Exodus
  []
[]

[MultiApps]
  [AForm]
    type = FullSolveMultiApp
    input_files = SubmeshAVFormSolve_Emod.i
    execute_on = timestep_begin
  []
[]

[Transfers]
  #transfer the calculated power from the initialisation Aform
  [pull_power]
    type = MultiAppMFEMTolibMeshShapeEvaluationTransfer
    from_multi_app = AForm
    source_variable = q_field
    variable = P
  []
  [Temp_to_MFEM]
    type = MultiApplibMeshToMFEMShapeEvaluationTransfer
    to_multi_app = AForm
    source_variable = T
    variable = temp
    execute_on = 'initial timestep_begin'
  []
[]
