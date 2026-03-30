vacuum_permeability  = ${fparse 4e-7*pi}                      # H/m
vacuum_reluctivity   = ${fparse 1/vacuum_permeability}        # (H/m)^-1
vacuum_econductivity = 1e-6                                   # S/m
vacuum_tconductivity = 0                                      # W/(m*K)
vacuum_density       = 0                                      # kg/m^3
vacuum_capacity      = 0                                      # J/(kg*K)

copper_permeability  = ${fparse vacuum_permeability}          # H/m
copper_reluctivity   = ${fparse 1/copper_permeability}        # (H/m)^-1
copper_econductivity = 5.96e7                                 # S/m
copper_econductivity_0 = 5.96e7                                 # S/m
copper_econd_alpha = 0.0039
copper_tconductivity = 400.59                                 # W/(m*K)
copper_density       = 8.94e3                                 # kg/m^3
copper_capacity      = 388.29                                 # J/(kg*K)

steel_permeability   = ${fparse vacuum_permeability}          # H/m
steel_reluctivity    = ${fparse 1/steel_permeability}         # (H/m)^-1
steel_reluctivity_0    = ${fparse 1/steel_permeability}         # (H/m)^-1
steel_econductivity  = 1.29e6                                 # S/m
steel_econductivity_0  = 1.29e6                                 # S/m
steel_econd_alpha = 0.006
steel_tconductivity  = 14.28                                  # W/(m*K)
steel_density        = 8.00e3                                # kg/m^3
steel_capacity       = 472.40                                 # J/(kg*K)
steel_tconductivity_0  = 14.28                                  # W/(m*K)
steel_density_0        = 8.00e3                                 # kg/m^3
steel_capacity_0       = 472.40                                 # J/(kg*K)

room_temperature     = 293.15                                 # K
t_initial            = 423.15                                 # K
Fluid_Temp           = 423.15

voltage_amplitude    = 0.2998340                              # V
voltage_frequency    = 1e5                                    # Hz
voltage_wfrequency   = ${fparse 2*pi*voltage_frequency}       # rad/s
voltage_period       = ${fparse 1/voltage_frequency}          # s

delta_t_af           = ${fparse voltage_period/50}            # s
skip_t_af            = ${fparse voltage_period}               # s
end_t_af             = ${fparse voltage_period*2}             # s

delta_t_th           = 0.05#0.5                                      # s
end_t_th             = 60                                     # s

visualization        = true

Re_block = ${fparse 919.131*7.36*0.012/0.00018551566535600087}
Pr_block = 1.170613923301386
Nu_block = ${fparse 0.023*pow(Re_block,0.8)*pow(Pr_block,0.4)}
HTC_block = ${fparse Nu_block*0.6062696884457435/0.012}