import re
import ast
import os
from typing import Dict, Any, Tuple, List
from Minigrid.utils.compare import compare_maps


Pair = Tuple[Dict[Any, Any], Dict[Any, Any]]  # (ground_truth, pred_map)


def detect_source(log_text: str, match_start: int) -> str:
    """
    Decide whether the matched GT/Map pair belongs to RBPF or LLM by
    scanning backward to the nearest marker.
    """
    last_rbpf = log_text.rfind("RBPF Result:", 0, match_start)
    last_llm_resp = log_text.rfind("LLM Response", 0, match_start)
    # Fallback: also consider a generic "Judgment" line after an LLM response block
    last_judgment = log_text.rfind("--- Judgment for Episode", 0, match_start)

    # Choose the nearest major marker among RBPF or LLM clues
    last_marker = max(last_rbpf, last_llm_resp, last_judgment)

    if last_marker == last_rbpf and last_rbpf != -1:
        return "RBPF"
    # Default to LLM when unsure
    return "LLM"


def extract_pairs_by_source(log_text: str) -> Dict[str, List[Pair]]:
    """
    Return {'RBPF': [(gt, pred), ...], 'LLM': [(gt, pred), ...]}.
    The regex tolerates either "LLM's Map" or "RBPF's Map" prefixes.
    """
    # Ground Truth and Map appear on single lines in your logs; be strict to avoid over-capture
    pattern = re.compile(
        r"Ground Truth:\s*(\{[^\n]*\})\s*\n.*?Map\s*:\s*(\{[^\n]*\})"
    )

    results = {"RBPF": [], "LLM": []}

    for m in pattern.finditer(log_text):
        src = detect_source(log_text, m.start())
        gt_str = m.group(1)
        mp_str = m.group(2)
        try:
            gt = ast.literal_eval(gt_str)
            mp = ast.literal_eval(mp_str)
            results[src].append((gt, mp))
        except Exception:
            # Skip malformed entries silently
            continue
    print(results)
    return results


def analyze_pairs(pairs: List[Pair]) -> Tuple[str, float, float, int]:
    """
    Build a detailed report string and compute averages.
    Returns (report_text, avg_overall, avg_goal, n_entries)
    """
    lines: List[str] = []
    overall_list: List[float] = []
    goal_list: List[float] = []

    for i, (gt, mp) in enumerate(pairs, 1):
        try:
            analysis = compare_maps(gt, mp)
            # 'details' is a multi-line string your helper already formats
            lines.append(f"================== Entry {i} ==================")
            lines.append(analysis["details"])
            lines.append("")
            overall_list.append(analysis["overall_accuracy"])
            goal_list.append(analysis["goal_accuracy"])
        except Exception as e:
            lines.append(f"================== Entry {i} ==================")
            lines.append(f"Analysis error: {e}")
            lines.append("")

    n = len(pairs)
    if n > 0:
        avg_overall = 100.0 * (sum(overall_list) / n)
        avg_goal = 100.0 * (sum(goal_list) / n)
        lines.append("================== FINAL SUMMARY ==================")
        lines.append(f"Analyzed {n} entries.")
        lines.append(f"Average Overall Accuracy (Jaccard): {avg_overall:.2f}%")
        lines.append(f"Average Goal-Only Accuracy        : {avg_goal:.2f}%")
        lines.append("")
        return "\n".join(lines), avg_overall, avg_goal, n
    else:
        msg = "No valid data entries found for this source."
        return msg, 0.0, 0.0, 0


def analyze_log_file_dual(input_file_path: str, base_output_path: str):
    """
    Read one log file, split pairs into RBPF vs LLM buckets, analyze each,
    and write two reports when possible.
    """
    try:
        print(f"Reading data from '{input_file_path}'...")
        with open(input_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"❌ Error: The file '{input_file_path}' was not found.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return

    buckets = extract_pairs_by_source(text)

    # RBPF
    rbpf_report, rbpf_overall, rbpf_goal, rbpf_n = analyze_pairs(buckets["RBPF"])
    if rbpf_n > 0:
        out_rbpf = f"{base_output_path}_RBPF_analysis_summary.txt"
        try:
            with open(out_rbpf, 'w', encoding='utf-8') as f:
                f.write(rbpf_report)
            print(f"✅ RBPF analysis saved to: {out_rbpf}")
        except IOError as e:
            print(f"❌ Could not write RBPF report: {e}")
    else:
        print("⚠️ No RBPF entries detected in this log.")

    # LLM
    llm_report, llm_overall, llm_goal, llm_n = analyze_pairs(buckets["LLM"])
    if llm_n > 0:
        out_llm = f"{base_output_path}_LLM_analysis_summary.txt"
        try:
            with open(out_llm, 'w', encoding='utf-8') as f:
                f.write(llm_report)
            print(f"✅ LLM analysis saved to: {out_llm}")
        except IOError as e:
            print(f"❌ Could not write LLM report: {e}")
    else:
        print("⚠️ No LLM entries detected in this log.")

    # Console combined summary
    print("-" * 70)
    print("Combined summary for this file:")
    print(f"  RBPF -> n={rbpf_n}, Avg Overall={rbpf_overall:.2f}%, Avg Goal={rbpf_goal:.2f}%")
    print(f"  LLM  -> n={llm_n}, Avg Overall={llm_overall:.2f}%, Avg Goal={llm_goal:.2f}%")
    print("-" * 70)


# --- MAIN ---
if __name__ == "__main__":

    # ▼▼▼ 1. CONFIGURE YOUR BATCH JOB HERE ▼▼▼
    ROOT_DIRECTORY = 'my_tests/8.24'
    TARGET_LOG_FILENAME = 'evaluation_log.txt'
    # This base suffix will be extended as *_RBPF_analysis_summary.txt and *_LLM_analysis_summary.txt
    REPORT_SUFFIX_BASE = '_analysis_summary'
    # --- END OF CONFIGURATION ---

    if not os.path.isdir(ROOT_DIRECTORY):
        print(f"❌ Error: Root directory '{ROOT_DIRECTORY}' not found. Please check the path.")
    else:
        print(f"🚀 Starting batch analysis in directory: '{ROOT_DIRECTORY}'")
        print(f"🔍 Searching for all files named: '{TARGET_LOG_FILENAME}'")
        print("-" * 70)

        files_found_count = 0

        for dirpath, _, filenames in os.walk(ROOT_DIRECTORY):
            if TARGET_LOG_FILENAME in filenames:
                files_found_count += 1
                input_file_path = os.path.join(dirpath, TARGET_LOG_FILENAME)
                base_name, _ = os.path.splitext(input_file_path)
                # Base path used to form two outputs:
                base_output_path = f"{base_name}{REPORT_SUFFIX_BASE}"

                print(f"Processing file #{files_found_count}: {input_file_path}")
                analyze_log_file_dual(input_file_path, base_output_path)
                print("-" * 70)

        if files_found_count == 0:
            print(f"⚠️ No files named '{TARGET_LOG_FILENAME}' were found in '{ROOT_DIRECTORY}'.")
        else:
            print(f"🎉 Batch processing complete. Analyzed {files_found_count} log file(s).")
