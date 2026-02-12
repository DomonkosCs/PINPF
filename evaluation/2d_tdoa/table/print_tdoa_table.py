import torch
import numpy as np
import pandas as pd
import os


def print_tdoa_table():
    # Load results dictionary from pt file
    pt_input_path = os.path.join(
        os.path.dirname(__file__), "tdoa_comparison_results.pt"
    )
    if not os.path.exists(pt_input_path):
        print(f"Error: {pt_input_path} does not exist.")
        return

    # Map location to cpu to avoid cuda errors if running on machine without gpu
    results = torch.load(pt_input_path, map_location=torch.device("cpu"))
    print(f"Loaded results from {pt_input_path}")

    # Create DataFrame
    df_data = []
    for name, metrics in results.items():
        if not metrics["ED"]:
            continue

        ed_mean = np.nanmean(metrics["ED"])
        swd_mean = np.nanmean(metrics["SWD"])
        time_mean = np.nanmean(metrics["Time"])

        df_data.append(
            {
                "Method": name,
                "ED Mean": ed_mean,
                "SWD Mean": swd_mean,
                "Time Mean": time_mean,
            }
        )

    df = pd.DataFrame(df_data)
    print("\nEvaluation Results (2D TDOA):")
    print(df.to_string(index=False))

    if len(df) > 0:
        first_method = df.iloc[0]["Method"]
        N_test = len(results[first_method]["ED"])
    else:
        N_test = 0

    print("\n" + r"\begin{table}[t]")
    print(r"    \centering")
    print(
        f"    \\caption{{Performance summary over {N_test} samples from the test dataset (mean values).}}"
    )
    print(r"    \label{tab:performance_summary}")
    print(r"        \begin{tabular}{@{}lccc@{}}")
    print(r"            \toprule")
    print(r"            Method & ED & SWD & Time [s] \\")
    print(r"            \midrule")

    for index, row in df.iterrows():
        method = row["Method"]
        ed_mean = row["ED Mean"]
        swd_mean = row["SWD Mean"]
        time_mean = row["Time Mean"]
        print(
            f"            {method:<25} & {ed_mean:.4f} & {swd_mean:.4f} & {time_mean:.4f} \\\\"
        )

    print(r"            \bottomrule")
    print(r"        \end{tabular}%")
    print(r"\end{table}")


if __name__ == "__main__":
    print_tdoa_table()
