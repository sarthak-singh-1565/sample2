
import runpy

print("\n===================================")
print("       SPOTIFY CHURN ANALYSIS")
print("===================================")

print("\nRunning Member 1: Data Analysis")
runpy.run_path("part1_data_analysis.py", run_name="__main__")

print("\nRunning Member 2: Logistic Regression")
runpy.run_path("part2_logistic_regression.py", run_name="__main__")

print("\nRunning Member 3: Linear Regression")
runpy.run_path("part3_linear_regression.py", run_name="__main__")

print("\nRunning Member 4: Final Integration")
runpy.run_path("part4_team_leader.py", run_name="__main__")

print("\n===================================")
print("       PROJECT COMPLETED")
print("===================================")
