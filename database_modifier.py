from database import get_recent_analyses
from database import get_db
db = next(get_db())


#query data
recent = get_recent_analyses(db, limit=100)
for analysis in recent:
    print(analysis.to_dict())