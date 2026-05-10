import os
import pandas as pd
import config

def generate_combined_excel():
    print("Generating combined Excel report...")
    
    # Load baseline
    baseline_df = pd.read_csv(os.path.join(config.RESULTS_DIR, "baseline_predictions.csv"))
    
    # Check column name for image path
    path_col = 'original_image_path' if 'original_image_path' in baseline_df.columns else 'image_path'
    
    # Map 0/1 to No/Yes for true label and baseline
    def map_label(val):
        return "yes" if val == 1 else "no"
        
    baseline_df['actual_answer'] = baseline_df['true_label'].apply(map_label)
    baseline_df['baseline_result'] = baseline_df['baseline_pred_label'].apply(map_label)
    
    # Keep only needed columns
    combined_df = baseline_df[[path_col, 'actual_answer', 'baseline_result']].copy()
    combined_df = combined_df.rename(columns={path_col: 'image_path'})
    
    # Load and merge each compression level
    for level in ['light', 'intermediate', 'aggressive']:
        pred_path = os.path.join(config.RESULTS_DIR, f"{level}_predictions.csv")
        if os.path.exists(pred_path):
            level_df = pd.read_csv(pred_path)
            
            # Create the yes/no column for this level
            col_name = f"{level}_compression_result"
            level_df[col_name] = level_df['pred_label'].apply(map_label)
            
            # Merge on original image path
            merge_df = level_df[['original_image_path', col_name]].rename(columns={'original_image_path': 'image_path'})
            combined_df = pd.merge(combined_df, merge_df, on='image_path', how='left')
            
    # Clean up image path for better readability (just filename)
    combined_df['image_filename'] = combined_df['image_path'].apply(os.path.basename)
    
    # Reorder columns
    cols = ['image_filename', 'actual_answer', 'baseline_result']
    for level in ['light', 'intermediate', 'aggressive']:
        col_name = f"{level}_compression_result"
        if col_name in combined_df.columns:
            cols.append(col_name)
            
    combined_df = combined_df[cols]
    
    # Define styling function
    def highlight_matches(row):
        colors = []
        actual = row['actual_answer']
        for col_name in row.index:
            if col_name.endswith('_result'):
                if pd.isna(row[col_name]):
                    colors.append('')
                elif str(row[col_name]).lower() == str(actual).lower():
                    colors.append('background-color: #C6EFCE; color: #006100') # Light green
                else:
                    colors.append('background-color: #FFC7CE; color: #9C0006') # Light red
            else:
                colors.append('')
        return colors
        
    # Save to Excel with formatting
    out_path = os.path.join(config.RESULTS_DIR, "combined_predictions_summary.xlsx")
    
    # Using pandas style engine which hooks directly to openpyxl
    styled_df = combined_df.style.apply(highlight_matches, axis=1)
    styled_df.to_excel(out_path, index=False)
    
    print(f"Excel file created successfully at: {out_path}")

if __name__ == "__main__":
    generate_combined_excel()