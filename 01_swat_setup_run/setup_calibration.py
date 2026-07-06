"""
SWAT+ calibration parameter injection tool
Functions:
1. Write calibration.cal into the TxtInOut directory
2. Modify file.cio to declare cal_parms.cal and calibration.cal on the chg line

Usage: change the TXTINOUT_DIR path below, then run.
"""

import shutil
from pathlib import Path

# ============================================================
# User configuration
# ============================================================
TXTINOUT_DIR = Path("./brazil/TxtInOut")  # change to your TxtInOut directory

# Parameters to adjust (add/remove/modify freely)
# Format: (param_name, change_type, change_value)
#   chg_typ: "pctchg" = percent change, "absval" = direct assignment, "abschg" = absolute add/subtract
PARAMS = [
    ("cn2",       "pctchg", -30.0),    # lowered further from -25 to -30
    ("alpha",     "absval",   0.008),
    ("deep_seep", "absval",   0.005),
    ("flo_min",   "absval",   0.50),
    ("revap_co",  "absval",   0.02),
    ("esco",      "absval",   0.50),
    ("surlag",    "absval",   24.0),
    ("latq_co",   "absval",   0.10),
    ("perco",     "absval",   0.80),
    ("cn3_swf",   "absval",   0.30),
    ("awc",       "pctchg",   50.0),   # increase soil water storage capacity by 50%
]

# ============================================================
# 1. Generate calibration.cal
# ============================================================
def write_calibration_cal(txtinout_dir, params):
    cal_path = txtinout_dir / "calibration.cal"
    
    with open(cal_path, 'w') as f:
        # Line 1: title
        f.write("calibration.cal: written by calibration setup script\n")
        # Line 2: number of parameters
        f.write(f"{len(params)}\n")
        # Line 3: column headers
        f.write(f"{'cal_parm':<24s}{'chg_typ':>12s}{'chg_val':>12s}"
                f"{'conds':>10s}{'soil_lyr1':>10s}{'soil_lyr2':>10s}"
                f"{'yr1':>10s}{'yr2':>10s}{'day1':>10s}{'day2':>10s}"
                f"{'obj_tot':>10s}\n")
        # data rows
        for name, chg_typ, chg_val in params:
            f.write(f"{name:<24s}{chg_typ:>12s}{chg_val:>12.5f}"
                    f"{'0':>10s}{'0':>10s}{'0':>10s}"
                    f"{'0':>10s}{'0':>10s}{'0':>10s}{'0':>10s}"
                    f"{'0':>10s}\n")
    
    print(f"✓ Written: {cal_path}")
    return cal_path


# ============================================================
# 2. Modify file.cio, adding the declaration on the chg line
# ============================================================
def update_file_cio(txtinout_dir):
    cio_path = txtinout_dir / "file.cio"

    if not cio_path.exists():
        raise FileNotFoundError(f"Could not find {cio_path}")

    # backup
    backup_path = txtinout_dir / "file.cio.bak"
    if not backup_path.exists():
        shutil.copy2(cio_path, backup_path)
        print(f"✓ Backed up: {backup_path}")
    
    with open(cio_path, 'r') as f:
        lines = f.readlines()
    
    # find the chg line and modify it
    modified = False
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) > 0 and parts[0] == 'chg':
            # chg line: must ensure both cal_parms.cal and calibration.cal are present
            # check the content of the current chg line
            print(f"  Original chg line (line {i+1}): {line.rstrip()}")

            # build the new chg line
            # file.cio format: keyword + several file names, null means an empty slot
            # chg line needs: cal_parms.cal  calibration.cal  followed by null padding
            new_chg_parts = ['chg']
            new_chg_parts.append('cal_parms.cal')
            new_chg_parts.append('calibration.cal')
            # pad with null (match the number of columns in the original line)
            n_fields = max(len(parts), 10)  # at least 10 columns
            while len(new_chg_parts) < n_fields:
                new_chg_parts.append('null')

            # format with fixed width (aligned with the other lines in file.cio)
            new_line = f"{new_chg_parts[0]:<20s}"
            for p in new_chg_parts[1:]:
                new_line += f"{p:<20s}"
            new_line += "\n"

            lines[i] = new_line
            modified = True
            print(f"  New chg line (line {i+1}): {new_line.rstrip()}")
            break

    if not modified:
        print("  Warning: chg line not found! Please check the file.cio format manually")
        return
    
    with open(cio_path, 'w') as f:
        f.writelines(lines)
    
    print(f"✓ Updated: {cio_path}")


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    print(f"TxtInOut directory: {TXTINOUT_DIR}\n")

    # check directory
    if not TXTINOUT_DIR.exists():
        print(f"Error: directory does not exist {TXTINOUT_DIR}")
        print("Please change the TXTINOUT_DIR path in the script and try again")
        exit(1)

    if not (TXTINOUT_DIR / "file.cio").exists():
        print(f"Error: file.cio not found, confirm that {TXTINOUT_DIR} is a TxtInOut directory")
        exit(1)

    # write calibration.cal
    write_calibration_cal(TXTINOUT_DIR, PARAMS)

    # modify file.cio
    update_file_cio(TXTINOUT_DIR)

    print("\nDone! You can re-run SWAT+ now.")
    print("To restore, simply replace file.cio with file.cio.bak.")
