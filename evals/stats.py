import json

def getAttackSuccessRate(filename):
    with open(filename, 'r') as f:
        result_list = json.load(f)
    top1Success = 0
    topKSuccess = 0
    baseSuccess = 0
    for result in result_list:
        if result["true_label_idx"] == result["original_top1_index"]:
            baseSuccess += 1
        if result["true_label_idx"] != result["topk_indices"][0]:
            top1Success += 1
        else:
            print(result["input_name"])
        if not result["true_label_idx"] in result["topk_indices"]:
            topKSuccess += 1
    print("Base success num:", baseSuccess)
    print("Base failed num:", len(result_list) - baseSuccess)
    print("Top 1 success num:", top1Success)
    print("Top K success num:", topKSuccess)
    baseSuccessRate = baseSuccess / len(result_list)
    top1SuccessRate = top1Success / len(result_list)
    topKSuccessRate = topKSuccess / len(result_list)
    print("Base success rate:", baseSuccessRate)
    print("Top 1 success rate:", top1SuccessRate)
    print("Top K success rate:", topKSuccessRate)
    return baseSuccessRate, top1SuccessRate, topKSuccessRate

if __name__ == "__main__":
    getAttackSuccessRate("./FGSM/FGSM_MobileViT.json")