import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def graph_data(csv):
    df_raw = pd.read_csv(csv)

    # String to dict and to df
    def parse_column(col):
        dicts = [ast.literal_eval(x) for x in col]
        return pd.DataFrame(dicts)

    df_struct = parse_column(df_raw['structured'])
    df_unstruct = parse_column(df_raw['unstructured'])

    df_struct = df_struct.rename(columns={'compression_real': 'Real%', 'f1_macro': 'F1-Macro'})
    df_unstruct = df_unstruct.rename(columns={'compression_real': 'Real%', 'f1_macro': 'F1-Macro'})

    pruning_levels = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,
                      0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    df_struct['Real%'] = [x * 100 for x in pruning_levels[:len(df_struct)]]
    df_unstruct['Real%'] = [x * 100 for x in pruning_levels[:len(df_unstruct)]]
    df_unstruct['Real%'] = [x * 100 for x in pruning_levels[:len(df_unstruct)]]
    df_unstruct['Real%'] = [x * 100 for x in pruning_levels[:len(df_unstruct)]]

    sns.set_theme(style="whitegrid", font_scale=1.1, palette="deep")

    sns.set_theme(style="whitegrid", font_scale=1.15)

    # --- 1) F1-Macro vs. Pruning Real ---
    plt.figure(figsize=(15, 8))
    sns.lineplot(x=df_struct['Real%'], y=df_struct['F1-Macro'], marker='o', label='Estructurado', color='royalblue')
    sns.lineplot(x=df_unstruct['Real%'], y=df_unstruct['F1-Macro'], marker='o', label='No estructurado',
                 color='darkorange')

    idx_s_struct = df_struct['F1-Macro'].idxmax()
    x_struct = df_struct.loc[idx_s_struct, 'Real%']
    y_struct = df_struct.loc[idx_s_struct, 'F1-Macro']
    plt.scatter(x_struct, y_struct, color='navy', s=130, zorder=5)
    plt.text(x_struct, y_struct - 0.01, f"{y_struct:.3f} @ {x_struct:.2f}%", color='navy',
             fontsize=12, fontweight='bold', ha='center', va='top', zorder=10)

    idx_s_unstruct = df_unstruct['F1-Macro'].idxmax()
    x_unstruct = df_unstruct.loc[idx_s_unstruct, 'Real%']
    y_unstruct = df_unstruct.loc[idx_s_unstruct, 'F1-Macro']
    plt.scatter(x_unstruct, y_unstruct, color='darkorange', s=130, zorder=5)
    plt.text(x_unstruct, y_unstruct + 0.01, f"{y_unstruct:.3f} @ {x_unstruct:.2f}%", color='darkorange',
             fontsize=12, fontweight='bold', ha='center', va='bottom', zorder=10)

    plt.xlabel('Pruning Real (%)')
    plt.ylabel('F1-Macro')
    plt.title('F1-Macro vs. Pruning Real')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- 2) F1-Score Benigna y Maligna vs. Pruning Real ---
    # --- Benigna ---
    plt.figure(figsize=(15, 8))
    sns.lineplot(x=df_struct['Real%'], y=df_struct['f1_benign'], marker='o', label='Estructurado',
                 color='royalblue')
    sns.lineplot(x=df_unstruct['Real%'], y=df_unstruct['f1_benign'], marker='o', label='No estructurado',
                 color='orange')

    # Sweet spot Estructurado Benigna
    x_sb_struct = df_struct.loc[df_struct['f1_benign'].idxmax(), 'Real%']
    y_sb_struct = df_struct['f1_benign'].max()
    plt.scatter(x_sb_struct, y_sb_struct, color='navy', s=120, zorder=5)
    plt.text(x_sb_struct, y_sb_struct + 0.03, f"{y_sb_struct:.3f} @ {x_sb_struct:.2f}%", color='navy', fontsize=11,
             fontweight='bold')

    # Sweet spot No estructurado Benigna
    x_sb_unstruct = df_unstruct.loc[df_unstruct['f1_benign'].idxmax(), 'Real%']
    y_sb_unstruct = df_unstruct['f1_benign'].max()
    plt.scatter(x_sb_unstruct, y_sb_unstruct, color='darkorange', s=120, zorder=5)
    plt.text(x_sb_unstruct, y_sb_unstruct + 0.045, f"{y_sb_unstruct:.3f} @ {x_sb_unstruct:.2f}%", color='darkorange',
             fontsize=11, fontweight='bold')

    plt.xlabel('Pruning Real (%)')
    plt.ylabel('F1 Score clase benigna')
    plt.title('F1-Score clase benigna vs. Pruning Real')
    plt.legend()
    plt.ylim(top=0.9)
    plt.tight_layout()
    plt.show()

    # --- Maligna ---
    plt.figure(figsize=(15, 8))
    sns.lineplot(x=df_struct['Real%'], y=df_struct['f1_malignant'], marker='s', label='Estructurado',
                 color='royalblue')
    sns.lineplot(x=df_unstruct['Real%'], y=df_unstruct['f1_malignant'], marker='s', label='No estructurado',
                 color='orange')

    # Sweet spot Estructurado Maligna
    x_sm_struct = df_struct.loc[df_struct['f1_malignant'].idxmax(), 'Real%']
    y_sm_struct = df_struct['f1_malignant'].max()
    plt.scatter(x_sm_struct, y_sm_struct, color='navy', s=120, zorder=5)
    plt.text(x_sm_struct, y_sm_struct + 0.04, f"{y_sm_struct:.3f} @ {x_sm_struct:.2f}%", color='navy', fontsize=11,
             fontweight='bold')

    # Sweet spot No estructurado Maligna
    x_sm_unstruct = df_unstruct.loc[df_unstruct['f1_malignant'].idxmax(), 'Real%']
    y_sm_unstruct = df_unstruct['f1_malignant'].max()
    plt.scatter(x_sm_unstruct, y_sm_unstruct, color='darkorange', s=120, zorder=5)
    plt.text(x_sm_unstruct, y_sm_unstruct + 0.04, f"{y_sm_unstruct:.3f} @ {x_sm_unstruct:.2f}%", color='darkorange',
             fontsize=11, fontweight='bold')

    plt.xlabel('Pruning Real (%)')
    plt.ylabel('F1 Score clase maligna')
    plt.title('F1-Score clase maligna vs. Pruning Real')
    plt.legend()
    plt.ylim(top=0.9)
    plt.tight_layout()
    plt.show()

    # --- 3) Accuracy vs. Pruning Real ---
    plt.figure(figsize=(15, 8))
    sns.lineplot(x=df_struct['Real%'], y=df_struct['accuracy'], marker='o', label='Estructurado', color='royalblue')
    sns.lineplot(x=df_unstruct['Real%'], y=df_unstruct['accuracy'], marker='o', label='No estructurado',
                 color='darkorange')

    # Calcular índices del sweet spot
    idx_s_struct = df_struct['accuracy'].idxmax()
    idx_s_unstruct = df_unstruct['accuracy'].idxmax()

    # Sweet spot Estructurado
    x_struct = df_struct.loc[idx_s_struct, 'Real%']
    y_struct = df_struct['accuracy'].max()
    plt.scatter(x_struct, y_struct, color='navy', s=130, zorder=5)
    plt.text(x_struct, y_struct + 0.0005, f"{y_struct:.3f} @ {x_struct:.2f}%", color='navy', fontsize=12,
             fontweight='bold', zorder=10)

    # Sweet spot No estructurado
    x_unstruct = df_unstruct.loc[idx_s_unstruct, 'Real%']
    y_unstruct = df_unstruct['accuracy'].max()
    plt.scatter(x_unstruct, y_unstruct, color='darkorange', s=130, zorder=5)
    plt.text(x_unstruct, y_unstruct + 0.01, f"{y_unstruct:.3f} @ {x_unstruct:.2f}%", color='darkorange', fontsize=12,
             fontweight='bold')

    plt.xlabel('Pruning Real (%)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Pruning Real')
    plt.legend()  # Sólo métodos, sin sweet spots
    plt.tight_layout()
    plt.show()

    # --- 4) Tamaño en disco vs. Pruning Real (barras logarítmicas) ---
    plt.figure(figsize=(15, 8))
    sns.barplot(x=df_struct['Real%'], y=df_struct['size_disk'], color='royalblue', label='Estructurado')
    sns.barplot(x=df_unstruct['Real%'], y=df_unstruct['size_disk'], color='orange', alpha=0.7, label='No estructurado')
    plt.yscale("log")
    plt.xlabel('Pruning Real (%)')
    plt.ylabel('Tamaño en disco (MB, log)')
    plt.title('Tamaño de modelo vs. Pruning Real (escala logarítmica)')
    plt.legend(["Estructurado", "No estructurado"])
    plt.tight_layout()
    plt.show()

    # F1-Macro vs. Active parameters
    plt.figure(figsize=(14, 8))
    sns.lineplot(x=df_struct["active_params"], y=df_struct["F1-Macro"], marker='o', label='Estructurado',
                 color='royalblue')
    sns.lineplot(x=df_unstruct["active_params"], y=df_unstruct["F1-Macro"], marker='o', label='No estructurado',
                 color='darkorange')

    # Sweet spot Estructurado
    idx_struct = df_struct["F1-Macro"].idxmax()
    x_struct = df_struct.loc[idx_struct, "active_params"]
    y_struct = df_struct.loc[idx_struct, "F1-Macro"]
    plt.scatter(x_struct, y_struct, color='navy', s=130, zorder=5)
    plt.text(x_struct, y_struct + 0.005, f"{y_struct:.3f} @ {int(x_struct)} params", color='navy', fontsize=12,
             fontweight='bold', ha='center')

    # Sweet spot No estructurado
    idx_unstruct = df_unstruct["F1-Macro"].idxmax()
    x_unstruct = df_unstruct.loc[idx_unstruct, "active_params"]
    y_unstruct = df_unstruct.loc[idx_unstruct, "F1-Macro"]
    plt.scatter(x_unstruct, y_unstruct, color='darkorange', s=130, zorder=5)
    plt.text(x_unstruct, y_unstruct + 0.005, f"{y_unstruct:.3f} @ {int(x_unstruct)} params", color='darkorange',
             fontsize=12, fontweight='bold', ha='center')

    plt.xlabel('Parámetros activos')
    plt.ylabel('F1-Macro')
    plt.title('F1-Macro vs. Número de parámetros activos')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # F1 clase minoritaria (maligna) vs. Número de parámetros activos
    plt.figure(figsize=(14, 8))
    sns.lineplot(x=df_struct["active_params"], y=df_struct["f1_malignant"], marker='s', label='Estructurado',
                 color='royalblue')
    sns.lineplot(x=df_unstruct["active_params"], y=df_unstruct["f1_malignant"], marker='s',
                 label='No estructurado', color='darkorange')

    # Sweet spot Estructurado Maligna
    idx_struct_mal = df_struct["f1_malignant"].idxmax()
    x_struct_mal = df_struct.loc[idx_struct_mal, "active_params"]
    y_struct_mal = df_struct.loc[idx_struct_mal, "f1_malignant"]
    plt.scatter(x_struct_mal, y_struct_mal, color='navy', s=120, zorder=5)
    plt.text(x_struct_mal, y_struct_mal + 0.01, f"{y_struct_mal:.3f} @ {int(x_struct_mal)} params", color='navy',
             fontsize=11, fontweight='bold', ha='center')

    # Sweet spot No estructurado Maligna
    idx_unstruct_mal = df_unstruct["f1_malignant"].idxmax()
    x_unstruct_mal = df_unstruct.loc[idx_unstruct_mal, "active_params"]
    y_unstruct_mal = df_unstruct.loc[idx_unstruct_mal, "f1_malignant"]
    plt.scatter(x_unstruct_mal, y_unstruct_mal, color='darkorange', s=120, zorder=5)
    plt.text(x_unstruct_mal, y_unstruct_mal + 0.01, f"{y_unstruct_mal:.3f} @ {int(x_unstruct_mal)} params",
             color='darkorange', fontsize=11, fontweight='bold', ha='center')

    plt.xlabel('Parámetros activos')
    plt.ylabel('F1 clase maligna')
    plt.title('F1 clase maligna vs. Número de parámetros activos')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # F1 clase mayoritaria (benigna) vs. Número de parámetros activos
    plt.figure(figsize=(14, 8))
    sns.lineplot(x=df_struct["active_params"], y=df_struct["f1_benign"], marker='o', label='Estructurado',
                 color='royalblue')
    sns.lineplot(x=df_unstruct["active_params"], y=df_unstruct["f1_benign"], marker='o', label='No estructurado',
                 color='darkorange')

    # Sweet spot Estructurado Benigna
    idx_struct_ben = df_struct["f1_benign"].idxmax()
    x_struct_ben = df_struct.loc[idx_struct_ben, "active_params"]
    y_struct_ben = df_struct.loc[idx_struct_ben, "f1_benign"]
    plt.scatter(x_struct_ben, y_struct_ben, color='navy', s=120, zorder=5)
    plt.text(x_struct_ben, y_struct_ben - 0.05, f"{y_struct_ben:.3f} @ {int(x_struct_ben)} params", color='navy',
             fontsize=11, fontweight='bold', ha='center', zorder=10, va='bottom')

    # Sweet spot No estructurado Benigna
    idx_unstruct_ben = df_unstruct["f1_benign"].idxmax()
    x_unstruct_ben = df_unstruct.loc[idx_unstruct_ben, "active_params"]
    y_unstruct_ben = df_unstruct.loc[idx_unstruct_ben, "f1_benign"]
    plt.scatter(x_unstruct_ben, y_unstruct_ben, color='darkorange', s=120, zorder=5)
    plt.text(x_unstruct_ben, y_unstruct_ben + 0.01, f"{y_unstruct_ben:.3f} @ {int(x_unstruct_ben)} params",
             color='darkorange', fontsize=11, fontweight='bold', ha='center')

    plt.xlabel('Parámetros activos')
    plt.ylabel('F1 clase benigna')
    plt.title('F1 clase beningna vs. Número de parámetros activos')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates graphics based on pruning data.")
    parser.add_argument(
        "--csv-path",
        type=str,
        help="CSV file path."
    )

    args = parser.parse_args()
    graph_data(args.csv_path)