import csv
import sys

# 定义高传染风险协议关键词
risk_patterns = {
    "GPL": ["GPL", "GNU General Public License"],
    "LGPL": ["LGPL", "Lesser General Public License"],
    "AGPL": ["AGPL", "Affero General Public License"]
}

def check_licenses(csv_path):
    found_risk = False
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pkg = row["Name"]
            ver = row["Version"]
            lic = row["License"]

            for risk_type, keywords in risk_patterns.items():
                for kw in keywords:
                    if kw in lic:
                        print(f"【发现{risk_type}风险】包: {pkg} | 版本: {ver} | 许可证: {lic}")
                        found_risk = True
    return found_risk

if __name__ == "__main__":
    csv_file = "licenses.csv"
    has_risk = check_licenses(csv_file)
    if has_risk:
        sys.exit(1)  # CI流水线中返回失败
    else:
        print("✅ 未检测到GPL/LGPL/AGPL传染性协议")
        sys.exit(0)