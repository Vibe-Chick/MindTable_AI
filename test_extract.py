from ai_engine.profile import extract_profile

answers = {
    "q1": "낯선 사람들이랑 있을 때 에너지가 더 나요. 여행 다니는 것도 좋아하고요.",
    "q2": "친구들이랑 얘기하면서 풀어요",
    "q3": "보드게임, 사진 찍기",
    "q4_choice": "분위기메이커"
}

profile = extract_profile(answers)
print(profile)
