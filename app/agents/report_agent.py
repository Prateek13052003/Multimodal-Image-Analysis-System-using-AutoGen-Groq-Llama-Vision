def report_agent(data):

    report = f"""
IMAGE REPORT

Scene: {data['scene']}
People: {data['number_of_people']}
Activity: {data['activity']}
Emotion: {data['emotion']}
Category: {data['category']}
Objects: {', '.join(data['objects'])}
"""

    return report