# modelpy
Chemistry Land-surface Atmosphere Soil Slab model | Python version

This is an extension of the z_t_flex branch (possible time and height dependent input). 
Added:
- diagnosis of RHtend factors
- cumulus model paramers phi_cu, wcld_prefact
- exner calculation is now based on reference pressure P_ref
- advection split up between free troposphere and mixed-layer
- compatible with plume module
- feedback of plume model on mass-flux
- free-tropospheric storage module
- alternative parametrization of acc
