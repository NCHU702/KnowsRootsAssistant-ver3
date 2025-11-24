
import pandas as pd

# 讀取你上傳的 CSV
df = pd.read_csv('export.csv')

# --- 步驟 1: 製作節點表 (Nodes) ---
# 整理 Source 節點 (通常是 Paper)
# 邏輯：取 id(n) 當 Id，優先取 n.title 當 Label (如果沒 title 就取 n.name)
source_nodes = df[['id(n)', 'n.title', 'n.name']].copy()
source_nodes['Label'] = source_nodes['n.title'].fillna(source_nodes['n.name'])
source_nodes = source_nodes[['id(n)', 'Label']].rename(columns={'id(n)': 'Id'})

# 整理 Target 節點 (通常是 Method, Domain 等)
# 邏輯：取 id(m) 當 Id，優先取 m.name 當 Label (如果沒 name 就取 m.title)
target_nodes = df[['id(m)', 'm.name', 'm.title']].copy()
target_nodes['Label'] = target_nodes['m.name'].fillna(target_nodes['m.title'])
target_nodes = target_nodes[['id(m)', 'Label']].rename(columns={'id(m)': 'Id'})

# 合併兩邊的節點，並刪除重複的 ID
all_nodes = pd.concat([source_nodes, target_nodes]).drop_duplicates(subset=['Id'])
# 填補任何還剩下的空值為 "Unknown"
all_nodes['Label'] = all_nodes['Label'].fillna('Unknown')

# 儲存節點表
all_nodes.to_csv('gephi_nodes_clean.csv', index=False, encoding='utf-8-sig')

# --- 步驟 2: 製作關係表 (Edges) ---
# 只需要 Source ID, Target ID 和 關係類型
edges = df[['id(n)', 'id(m)', 'type(r)']].copy()
edges.columns = ['Source', 'Target', 'Label']
edges['Type'] = 'Directed'  # Gephi 需要知道是有向圖

# 儲存關係表
edges.to_csv('gephi_edges_clean.csv', index=False, encoding='utf-8-sig')

print(f"處理完成！\n總共產出 {len(all_nodes)} 個節點，{len(edges)} 條關係。")