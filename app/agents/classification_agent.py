def classification_agent(vision_result):

    people = vision_result["number_of_people"]

    if people > 1:
        category = "Group Photo"
    else:
        category = "Single Person"

    vision_result["category"] = category

    return vision_result