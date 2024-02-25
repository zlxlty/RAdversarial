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
        
            if not result["true_label_idx"] in result["topk_indices"]:
                topKSuccess += 1
    print("Base success num:", baseSuccess)
    print("Base failed num:", len(result_list) - baseSuccess)
    print("Top 1 success num:", top1Success)
    print("Top K success num:", topKSuccess)
    baseSuccessRate = baseSuccess / len(result_list)
    top1SuccessRate = top1Success / baseSuccess
    topKSuccessRate = topKSuccess / baseSuccess
    print("Base success rate:", baseSuccessRate)
    print("Top 1 success rate:", top1SuccessRate)
    print("Top K success rate:", topKSuccessRate)
    return baseSuccessRate, top1SuccessRate, topKSuccessRate

if __name__ == "__main__":
    attacks = ["FGSM"]
    models = ["MobileViT", "ResNet50", "Surrogate"]
    for attack in attacks:
        for model in models:
            for i in range(0, 7):
                try:
                    filename = f"./{attack}_new/{attack}_{model}_{2**i}.0ep.json"
                    print(f"Attack: {attack}, Model: {model}, Ep: {2**i}")
                    getAttackSuccessRate(filename)
                    print()
                except:
                    continue
