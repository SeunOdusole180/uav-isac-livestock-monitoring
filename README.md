# UAV-Enabled ISAC Livestock Monitoring

UAV-enabled ISAC framework for large-scale livestock monitoring
using ns-3. Four heterogeneous UAVs simultaneously perform
radar-based livestock detection and transmit data via 5G NR.
Features offline deep RL optimisation over 300+ simulation runs
and an O-RAN xApp deployed in the Near-RT RIC for near-real-time
adaptive UAV sensing control.

**Thesis:** UAV-Enabled Integrated Sensing and Communication
for Large-Scale Livestock Monitoring
**Author:** Oluwaseun Odusole
**Institution:** University College Dublin, ME Electronic and
Computer Engineering, April 2026
**Supervisor:** Dr. Pasika Ranaweera

---

## File Descriptions

### ns-3 C++ Simulation Files

| File | Description |
|---|---|
| `uav-optimisation.cc` | Core 2D simulation scenario. Implements the full UAV-enabled ISAC framework with four heterogeneous UAV roles (Surveillance, Patrol, Rapid-Response, Strategic), monostatic radar sensing, SNR-weighted cooperative position fusion, 5G NR cellular uplink, and three-component energy model. Generates `run_summary.csv`. |
| `uav-optimisation-3d.cc` | Extended 3D terrain simulation. Identical to the 2D scenario but adds sinusoidal terrain elevation modelling and per-role sensing metrics including detection rate, RMSE, coverage, duty cycle, and energy per detection. Generates `run_summary_3d.csv`. |
| `uav-optimisation-xapp.cc` | xApp-enabled simulation. Extends the scenario with the `KpiReporter` class that emulates the O-RAN E2 interface, transmitting live KPI reports to the xApp over UDP and receiving parameter updates at 10-second intervals. Generates `xapp_kpi_log.csv`. |
| `uav-mobility.cc` | Standalone UAV mobility model development and testing script. Implements and validates the Enhanced Gauss-Markov mobility model across all four UAV roles before integration into the full simulation. |

### Python Analysis and Optimisation Files

| File | Description |
|---|---|
| `offline_rl_analysis.py` | Offline RL optimisation for the 2D dataset. Loads `run_summary.csv`, trains a Random Forest regressor to approximate the J-score objective function, and generates improved candidate parameter configurations for subsequent ns-3 simulation runs. Saves the trained model as `offline_rf_model.pkl`. |
| `offline_rl_analysis_3d.py` | Offline RL optimisation for the 3D dataset. Identical procedure to `offline_rl_analysis.py` but operates on `run_summary_3d.csv` and accounts for the extended per-role metrics available in the 3D dataset. |
| `xapp_bridge.py` | O-RAN xApp Python process. Implements the Near-RT RIC xApp with a UDP receiver thread for incoming KPI reports and a parameter update transmitter. Combines a five-rule threshold-based control policy (Pdet, RMSE, energy, delay, loss) with a model-guided Random Forest refinement component that activates every fifth control cycle. |
| `results_analysis_complete.py` | Complete results analysis script. Processes all three datasets (2D, 3D, xApp) to generate the full set of thesis figures including optimisation progress plots, scatter plots, boxplots, per-role bar charts, correlation heatmaps, xApp KPI time series, and the three-way comparison centrepiece figure. Outputs to `thesis_figures/`. |
| `results_tables_graphs.py` | Supplementary results script. Generates additional tables and graphs for specific thesis sections including parameter sensitivity analysis, feature importance plots, and energy breakdown figures. |


## Output Files

| File | |
|---|---|
| `run_summary.csv` | |
| `run_summary_3d.csv` | |
| `xapp_kpi_log.csv` | |
| `offline_rf_model.pkl` | |
| `thesis_figures/` | |

---

## Requirements

- ns-3 (version 3-dev) with CTTC nr module
- Python 3.11+
