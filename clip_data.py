import json
import random

def resplit_data():
    """重新分割数据为训练集和验证集"""
    
    # 读取原始数据
    with open('data.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"原始数据总行数: {len(lines)}")
    
    # 过滤空行和无效行
    valid_lines = []
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if line:  # 跳过空行
            try:
                # 验证JSON格式
                json.loads(line)
                valid_lines.append(line)
            except json.JSONDecodeError as e:
                print(f"跳过第 {i} 行（JSON格式错误）: {e}")
                continue
    
    print(f"有效数据行数: {len(valid_lines)}")
    
    if len(valid_lines) < 10:
        print("错误: 有效数据不足10行")
        return
    
    # 随机打乱数据
    random.shuffle(valid_lines)
    
    # 分割数据（10条作为验证集）
    val_lines = valid_lines[:10]
    train_lines = valid_lines[10:]
    
    # 保存验证集
    with open('val_data.jsonl', 'w', encoding='utf-8') as f:
        for line in val_lines:
            f.write(line + '\n')
    
    # 保存训练集
    with open('train_data.jsonl', 'w', encoding='utf-8') as f:
        for line in train_lines:
            f.write(line + '\n')
    
    print("✓ 数据分割完成！")
    print(f"训练集: {len(train_lines)} 行")
    print(f"验证集: {len(val_lines)} 行")
    
    # 验证生成的文件
    verify_files()

def verify_files():
    """验证生成的文件"""
    print("\n验证文件...")
    
    for file_path in ['train_data.jsonl', 'val_data.jsonl']:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                valid_count = 0
                for i, line in enumerate(lines, 1):
                    line = line.strip()
                    if line:
                        try:
                            json.loads(line)
                            valid_count += 1
                        except json.JSONDecodeError:
                            print(f"  {file_path} 第 {i} 行格式错误")
                
                print(f"  {file_path}: {valid_count} 行有效数据")
                
        except Exception as e:
            print(f"  {file_path}: 读取失败 - {e}")

if __name__ == "__main__":
    resplit_data()