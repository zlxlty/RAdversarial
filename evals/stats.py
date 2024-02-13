import json

def getAttackSuccessRate(filename):
    with open(filename, 'r') as f:
        result_list = json.load(f)
    top1Success = 0
    topKSuccess = 0
    for result in result_list:
        if result["true_label_idx"] != result["topk_indices"][0]:
            top1Success += 1
        else:
            print(result["input_name"])
        if not result["true_label_idx"] in result["topk_indices"]:
            topKSuccess += 1
    print("Top 1 success rate:", top1Success / len(result_list))
    print("Top K success rate:", topKSuccess / len(result_list))
    return top1Success / len(result_list), topKSuccess / len(result_list)

if __name__ == "__main__":
    getAttackSuccessRate("./PGD/PGD_MobileViT.json")