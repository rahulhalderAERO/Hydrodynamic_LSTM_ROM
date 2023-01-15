# Hydrodynamic_LSTM_ROM

- **Hydrodynamic_PINN.ipynb** is the LSTM-PINN file with the hydrodynamic forces as input and box responses as output and the rigid body dynamics equations are introduced in the loss function in the discretized form of the ODE. **LSTM_PINN_Data** contains the input output files of LSTM-PINN.
- **Hydrodynamic_DEIM_LSTM.ipynb** is the DEIM-LSTM algorithm without any equation based loss function where the wave height is the input and the box responses are the output.**DEIM_LSTM_Data** contains the input output files of DEIM-LSTM algorihm
- **DEIM folder** contains the DEIM algorithm implemented in MatLAB.
