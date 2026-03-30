# AV-Form frequency-domain solve
# https://doc.comsol.com/6.1/docserver/#!/com.comsol.help.acdc/acdc_ug_theory.05.51.html

# AC current frequency
freq = 1.23476e5 #1e5 # 100 kHz
angfreq = '${fparse 2.0*pi*freq}'

# Permittivity of free space
epsilon0 = 8.8541878176e-12

# Conductivities
T_avg = 450
sigma_coil = 3.3e7 #3.3e7#5.96e7 # S/m
sigma_vac = 0.0
sigma_target = 1.29e6*(1/(1 + )) #6.68e5#6.68e5#1e6

# Magnetic reluctivity of free space (1/mu0)
nu0 = '${fparse (1.0e7)/(4*pi)}'

potential_difference = 85 # V testing
coil_current = 2191.97 # A peak-to-peak
terminal_area = 2.4e-5 # m^2 for vac_oval_coil_solid_target_coarse.e coil
coil_av_current_density = '${fparse coil_current / terminal_area}'

# coil_in = 'coil_in'
# coil_out = 'coil_out'
# target = 'target'
# pipe_inner = 'pipe_inner'
# mesh = ../Remesh_in.e
# insulation = 'terminal_plane exterior1 exterior2 exterior3 exterior4 exterior5

coil_in = 'voltage-surf-2'
coil_out = 'voltage-surf-1'
target = 'monoblock'
pipe_inner = 'monoblock_htc'
mesh = '../vac_hive.e'
insulation = 'magnetic_insulation'


[Problem]
  type = MFEMProblem
  numeric_type = complex
[]

[Mesh]
  type = MFEMMesh
  file = ${mesh}
  #uniform_refine =1
[]

[SubMeshes]
  [coil_submesh]
    type = MFEMDomainSubMesh
    block = 'coil'
  []
[]

[FESpaces]
  [H1FESpace]
    type = MFEMScalarFESpace
    fec_type = H1
    fec_order = FIRST
    submesh = coil_submesh
  []
  [CoilHCurlFESpace]
    type = MFEMVectorFESpace
    fec_type = ND
    fec_order = FIRST
    submesh = coil_submesh
  []
  [HCurlFESpace]
    type = MFEMVectorFESpace
    fec_type = ND
    fec_order = FIRST
  []
  [HDivFESpace]
    type = MFEMVectorFESpace
    fec_type = RT
    fec_order = CONSTANT
  []
  [L2FESpace]
    type = MFEMScalarFESpace
    fec_type = L2
    fec_order = FIRST
  []
[]

[Variables]
  [a_field] # Magnetic vector potential A = iE_ind/w associated with induced electric field
    type = MFEMComplexVariable
    fespace = HCurlFESpace
  []
  [coil_electric_potential] # Electric potential associated with source electric field
    type = MFEMComplexVariable
    fespace = H1FESpace
  []
[]

[AuxVariables]
  [grad_v] # Electric potential gradient on coil submesh
    type = MFEMComplexVariable
    fespace = CoilHCurlFESpace
  []
  [parent_grad_v] #  Electric potential gradient on global mesh
    type = MFEMComplexVariable
    fespace = HCurlFESpace
  []
  [e_field] # total complex electric field E = E_ind + E_ext
    type = MFEMComplexVariable
    fespace = HCurlFESpace
  []
  [b_field] # complex magnetic flux density
    type = MFEMComplexVariable
    fespace = HDivFESpace
  []
  [q_field] # Joule heating on target
    type = MFEMVariable
    fespace = L2FESpace
  []
  [q1_field] # Joule heating on target
    type = MFEMVariable
    fespace = L2FESpace
  []
  [q2_field] # Joule heating on target
    type = MFEMVariable
    fespace = L2FESpace
  []
[]

[Functions]
  # (iωσ-ω²ε)A represented as (massCoef + i*loss_coef)A where massCoef = -ω²ε, lossCoef = ωσ
  [mass_coef]
    type = ParsedFunction
    expression = -${epsilon0}*${angfreq}^2
  []
  [loss_coef_vac]
    type = ParsedFunction
    expression = ${angfreq}*${sigma_vac}
  []
  [loss_coef_coil]
    type = ParsedFunction
    expression = ${angfreq}*${sigma_coil}
  []
  [loss_coef_target]
    type = ParsedFunction
    expression = ${angfreq}*${sigma_target}
  []
  [sigma_coil]
    type = ParsedFunction
    expression = ${sigma_coil}
  []
[]

[FunctorMaterials]
  [vacuum]
    type = MFEMGenericFunctorMaterial
    prop_names = 'massCoef lossCoef sigma nu'
    prop_values = 'mass_coef loss_coef_vac ${sigma_vac} ${nu0}'
    block = 'vacuum_region'
  []
  [coil]
    type = MFEMGenericFunctorMaterial
    prop_names = 'massCoef lossCoef sigma nu'
    prop_values = 'mass_coef loss_coef_coil sigma_coil ${nu0}'
    block = 'coil'
  []
  [target]
    type = MFEMGenericFunctorMaterial
    prop_names = 'massCoef lossCoef sigma nu'
    prop_values = 'mass_coef loss_coef_target ${sigma_target} ${nu0}'
    block = ${target}
  []
[]

[AuxKernels]
  [∇×A]
    type = MFEMComplexCurlAux
    variable = b_field
    source = a_field
    execute_on = TIMESTEP_END
  []
  [∇V]
    type = MFEMComplexGradAux
    variable = grad_v
    source = coil_electric_potential
    execute_on = TIMESTEP_END
  []
  [e_field] # E = -grad V - iwA
    type = MFEMComplexSumAux
    variable = e_field
    source_variables = 'parent_grad_v a_field'
    scale_factors_real = '-1.0 0.0'
    scale_factors_imag = '0.0 -${angfreq}'
    execute_on = TIMESTEP_END
    execution_order_group = 3 # Evaluate after transfer of grad v to parent mesh
  []
  [joule_heat_1]
    type = MFEMInnerProductAux
    variable = q1_field
    first_source_vec = e_field_real
    second_source_vec = e_field_real
    coefficient = sigma
    execute_on = TIMESTEP_END
    execution_order_group = 4 # Evaluate after e_field update
  []
  [joule_heat_2]
    type = MFEMInnerProductAux
    variable = q2_field
    first_source_vec = e_field_imag
    second_source_vec = e_field_imag
    coefficient = sigma
    execute_on = TIMESTEP_END
    execution_order_group = 4 # Evaluate after e_field update
  []
  [joule_heat]
    type = MFEMSumAux
    variable = q_field
    source_variables = 'q1_field q2_field'
    scale_factors = '0.5 0.5'
    execute_on = TIMESTEP_END
    execution_order_group = 5
  []
[]

[BCs]
  active = 'coil_voltage_constraint coil_ground exterior_a_field' # Strongly constrain coil voltage
  # active = 'coil_current_constraint coil_ground exterior_a_field' # Weakly constrain coil current density
  [exterior_a_field]
    type = MFEMComplexVectorTangentialDirichletBC # Enforces J normal to surface, B tangential to surface
    variable = a_field
    boundary = '${coil_out} ${coil_in} ${insulation} coil_htc'
  []
  [coil_ground]
    type = MFEMComplexScalarDirichletBC
    variable = coil_electric_potential
    boundary = '${coil_in}'
    coefficient_real = 0.0
    coefficient_imag = 0.0
  []
  [coil_voltage_constraint]
    type = MFEMComplexScalarDirichletBC
    variable = coil_electric_potential
    boundary = '${coil_out}'
    coefficient_real = '${potential_difference}'
    coefficient_imag = 0.0 #no phase-shift
  []
  [coil_current_constraint]
    type = MFEMComplexIntegratedBC
    variable = coil_electric_potential
    boundary = '${coil_in}'
    [RealComponent]
      type = MFEMBoundaryIntegratedBC
      coefficient = '${coil_av_current_density}'
    []
    [ImagComponent]
      type = MFEMBoundaryIntegratedBC
      coefficient = 0.0
    []
  []
[]

[Kernels]
  [ν∇×A,∇×A']
    type = MFEMComplexKernel
    variable = a_field
    [RealComponent]
      type = MFEMCurlCurlKernel
      coefficient = nu
      block = '${target} vacuum_region coil'
    []
  []
  [(iωσ-ω²ε)A,A']
    type = MFEMComplexKernel
    variable = a_field
    [RealComponent]
      type = MFEMVectorFEMassKernel
      coefficient = massCoef # = -ω²ε
      block = '${target} vacuum_region coil'
    []
    [ImagComponent]
      type = MFEMVectorFEMassKernel
      coefficient = lossCoef # = ωσ
      block = '${target} coil'
    []
  []
  [(σ+iωε)∇V,A']
    type = MFEMMixedSesquilinearFormKernel
    trial_variable = coil_electric_potential
    variable = a_field
    [RealComponent]
      type = MFEMMixedVectorGradientKernel
      coefficient = sigma_coil
    []
    [ImagComponent]
      type = MFEMMixedVectorGradientKernel
      coefficient = '${fparse angfreq * epsilon0}'
    []
  []

  # div J = 0 gauge choice in coil
  [σ∇V,∇V']
    type = MFEMComplexKernel
    variable = coil_electric_potential
    [RealComponent]
      type = MFEMDiffusionKernel
      coefficient = sigma_coil
    []
  []
  [iωσA,∇V']
    type = MFEMMixedSesquilinearFormKernel
    trial_variable = a_field
    variable = coil_electric_potential
    transpose = true
    [ImagComponent]
      type = MFEMMixedVectorGradientKernel
      coefficient = loss_coef_coil
    []
  []
[]

[Solver]
  type = MFEMMUMPS
[]

[Executioner]
  type = MFEMSteady
  device = cpu
[]

[Transfers]
  [submesh_transfer_from_coil]
    type = MFEMSubMeshComplexTransfer
    from_variable = grad_v
    to_variable = parent_grad_v
    execute_on = TIMESTEP_END
    execution_order_group = 2
  []
[]

[Outputs]
  [ParaViewDataCollection]
    type = MFEMParaViewDataCollection
    file_base = Submesh_AVform_frequency_domain
  []
  [SubmeshParaViewDataCollection]
    type = MFEMParaViewDataCollection
    file_base = CoilSubmesh_AVform_frequency_domain
    submesh = coil_submesh
  []
[]
