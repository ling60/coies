import root_path


volcab = set()
with open(root_path.T2T_DATA_DIR + root_path.T2T_AAER_VOLCAB_NAME, 'r') as f:
    for line in f:
        volcab.add(line)


with open(root_path.T2T_DATA_DIR + root_path.T2T_AAER_VOLCAB_NAME + '.40000', 'w') as f:
    for word in volcab:
        f.write(str(word) + '\n')

