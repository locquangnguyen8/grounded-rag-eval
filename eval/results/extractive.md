# Evaluation report (extractive)

| Metric | Value |
|---|---|
| cases | 36 |
| retrieval_recall_at_k | 1.0 |
| mrr | 0.9833 |
| answer_accuracy | 0.8333 |
| fact_coverage | 0.8333 |
| citation_validity | 1.0 |
| correct_abstention_rate | 0.8333 |
| answered_rate | 1.0 |
| latency_p50_ms | 0.19 |
| latency_p95_ms | 0.26 |
| input_tokens | 0 |
| output_tokens | 0 |

## Failed cases (6)

- **q01** (missing facts): When did the Harbourline Estuary Monitoring Program start operating?  
  Answer: The Harbourline Estuary Monitoring Program (HEMP) is a fictional water-quality monitoring program used as the knowledge base for this demo. [program_overview#0]
- **q13** (missing facts): How much rainfall triggers a harvest area closure?  
  Answer: The regional food authority closes a harvest area when either trigger is met: [harvest_closure_rules#0] salinity at the harvest-area station stays below 22 PSU for two consecutive days. [harvest_closure_rules#0]
- **q14** (missing facts): What salinity level triggers a closure?  
  Answer: Registered shellfish growers also receive SMS alerts when a closure trigger is met or a dissolved oxygen alert is raised. [stakeholder_reporting#0] It is not used directly as a closure trigger. [satellite_data#0]
- **q17** (missing facts): What type of model is used to forecast closure risk?  
  Answer: The bulletin summarises the previous week's readings, current closures and the 7-day closure risk forecast. [stakeholder_reporting#0] SHAP values are published alongside each forecast so growers can see which features drove the risk score. [forecasting_model#1]
- **q25** (missing facts): How long did the Merrin closure last after the March 2024 flood?  
  Answer: In March 2024, 148 mm of rain fell on the Merrin catchment over three days. [flood_event_2024#0]
- **u05** (should abstain): Who is the current program manager?  
  Answer: The program is run by a small team: one program manager, two field technicians and one data engineer. [program_overview#1] Alerts go to the program manager and to registered growers in the affected estuary. [dissolved_oxygen_alerts#0]

## Release gate

PASS
