import io
import re

import pdfplumber as pb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# weights: 80% skill match, 20% text similarity (same as your original)
SKILL_WEIGHT = 0.8
TFIDF_WEIGHT = 0.2


# accepts a file path OR a file-like object (needed for web uploads)
def extract(pdf_path) -> str:
    full_text = []
    with pb.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
    return "\n".join(full_text)


def extract_bytes(data: bytes) -> str:
    return extract(io.BytesIO(data))


# cleaning text
def clean_text(text: str, remove_email: bool = True, remove_url: bool = True,
               remove_extraspace: bool = True) -> str:
    # URLs and emails must be removed BEFORE stripping symbols,
    # otherwise ':' '/' '@' are already gone and the patterns never match
    if remove_url:
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    if remove_email:
        text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9+#./\- ]", " ", text)
    if remove_extraspace:
        text = re.sub(r"\s+", " ", text).strip()
    return text


# skills dictionary
SKILL_ALIASES = {
    # PROGRAMMING LANGUAGES
    "python": ["python", "python3", "python 3"],
    "java": ["java", "java se", "java ee"],
    "c++": ["c++", "cpp", "c plus plus"],
    "c": ["c programming", "c language"],
    "c#": ["c#", "c sharp", "csharp"],
    "javascript": ["javascript", "java script", "js", "ecmascript"],
    "julia": ["julia"],
    "typescript": ["typescript", "type script", "ts"],
    "go": ["golang", "go language", "go programming"],
    "rust": ["rust", "rust lang", "rust programming"],

    # WEB DEVELOPMENT
    "html": ["html", "html5", "hypertext markup language"],
    "css": ["css", "css3", "cascading style sheets"],
    "react": ["react", "reactjs", "react.js", "react js", "react library"],
    "angular": ["angular", "angularjs", "angular.js"],
    "vue": ["vue", "vuejs", "vue.js"],
    "nodejs": ["node", "nodejs", "node.js", "node js"],
    "express": ["express", "expressjs", "express.js", "express js"],
    "django": ["django", "django framework"],
    "flask": ["flask", "flask framework"],
    "fastapi": ["fastapi", "fast api"],

    # DATABASES
    "sql": ["sql", "structured query language"],
    "mysql": ["mysql", "my sql"],
    "postgresql": ["postgresql", "postgres", "postgre sql", "psql"],
    "mongodb": ["mongodb", "mongo db", "mongo"],
    "redis": ["redis", "redis database"],
    "oracle": ["oracle", "oracle database", "oracle db"],
    "database": ["database", "databases", "db"],

    # VERSION CONTROL
    "git": ["git", "git version control"],
    "github": ["github", "git hub"],
    "gitlab": ["gitlab", "git lab"],
    "bitbucket": ["bitbucket", "bit bucket"],

    # CONTAINERS / DEVOPS
    "docker": ["docker", "docker containers", "containerization", "containerisation"],
    "kubernetes": ["kubernetes", "k8s", "kube"],
    "jenkins": ["jenkins", "jenkins ci"],
    "ci/cd": ["ci/cd", "ci cd", "continuous integration",
              "continuous delivery", "continuous deployment"],

    # CLOUD
    "aws": ["aws", "amazon web services", "amazon aws"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "cloud computing": ["cloud computing", "cloud", "cloud technology", "cloud technologies"],

    # COMPUTER SCIENCE
    "data structures": ["data structures", "data structure", "ds", "dsa",
                        "data structures and algorithms"],
    "algorithms": ["algorithms", "algorithm", "algo", "dsa",
                   "data structures and algorithms"],
    "object oriented programming": ["object oriented programming",
                                    "object-oriented programming",
                                    "object oriented", "oop", "o o p"],
    "operating systems": ["operating systems", "operating system", "os", "o.s."],
    "computer networks": ["computer networks", "computer network",
                          "networking", "network", "cn"],
    "dbms": ["dbms", "database management system",
             "database management systems", "database systems"],
    "software engineering": ["software engineering", "software development",
                             "software engineering principles"],

    # APIs
    "rest api": ["rest api", "rest apis", "restful api", "restful apis",
                 "rest service", "rest services", "rest architecture",
                 "representational state transfer"],
    "api": ["api", "apis", "application programming interface",
            "application programming interfaces"],
    "graphql": ["graphql", "graph ql"],

    # AI / ML
    "machine learning": ["machine learning", "machine-learning", "ml",
                         "machine learning algorithms"],
    "deep learning": ["deep learning", "deep-learning", "dl", "deep neural networks"],
    "artificial intelligence": ["artificial intelligence",
                                "artificial intelligence and machine learning",
                                "ai", "a.i."],
    "natural language processing": ["natural language processing",
                                    "natural-language processing", "nlp", "n.l.p."],
    "computer vision": ["computer vision", "image processing", "image recognition"],
    "tensorflow": ["tensorflow", "tensor flow"],
    "pytorch": ["pytorch", "py torch"],
    "scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "pandas": ["pandas", "python pandas"],
    "numpy": ["numpy", "num py"],

    # DATA SCIENCE
    "data science": ["data science", "data sciences", "data scientist"],
    "data analysis": ["data analysis", "data analytics", "data analyst",
                      "data analysis techniques"],
    "statistics": ["statistics", "statistical analysis", "statistical modeling",
                   "statistical modelling"],
    "data visualization": ["data visualization", "data visualisation", "data viz",
                           "visualization", "visualisation"],

    # LINUX / TERMINAL
    "linux": ["linux", "gnu linux", "linux operating system"],
    "bash": ["bash", "bash scripting", "shell scripting", "shell script"],
    "unix": ["unix", "unix systems", "unix operating system"],

    # JAVA BACKEND
    "spring": ["spring", "spring framework"],
    "spring boot": ["spring boot", "springboot", "spring-boot"],
    "hibernate": ["hibernate", "hibernate orm"],

    # MOBILE
    "android": ["android", "android development", "android sdk"],
    "flutter": ["flutter", "flutter framework"],
    "react native": ["react native", "react-native", "reactnative"],

    # TESTING
    "unit testing": ["unit testing", "unit test", "unit tests"],
    "selenium": ["selenium", "selenium webdriver", "selenium testing"],
    "testing": ["software testing", "software test", "testing",
                "quality assurance", "qa"],

    # AGILE
    "agile": ["agile", "agile methodology", "agile methodologies", "agile development"],
    "scrum": ["scrum", "scrum methodology", "scrum framework"],

    # PROJECT MANAGEMENT
    "jira": ["jira", "atlassian jira"],
    "trello": ["trello"],
}

# Compiled once. Plain \b fails for c++ / c# / ci/cd / o.s. because '+', '#',
# '/' and '.' are not word characters, so lookarounds are used instead.
SKILL_PATTERNS = {
    skill: re.compile(
        r"(?<![\w+#])(?:" + "|".join(re.escape(v) for v in variants) + r")(?![\w+#])"
    )
    for skill, variants in SKILL_ALIASES.items()
}


# tfidf similarity
def tfidf_raw(text1: str, text2: str) -> float:
    if not text1.strip() or not text2.strip():
        return 0.0
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 3),
                          max_features=20000, sublinear_tf=True)
    try:
        matrix = vec.fit_transform([text1, text2])
    except ValueError:  # text contained only stop words
        return 0.0
    return float(cosine_similarity(matrix[0], matrix[1])[0][0])


def tfidfsim(text1: str, text2: str) -> float:
    return tfidf_raw(text1, text2) * TFIDF_WEIGHT


# skill matching
def find_skill(text: str):
    text = text.lower()
    return {skill for skill, pat in SKILL_PATTERNS.items() if pat.search(text)}


# calculating skill match score
def skill_ratio(cv: str, req: str) -> float:
    skillsfromreq = find_skill(req)
    if not skillsfromreq:  # avoids ZeroDivisionError
        return 0.0
    return len(find_skill(cv) & skillsfromreq) / len(skillsfromreq)


def skillmatchscore(cv: str, req: str) -> float:
    return skill_ratio(cv, req) * SKILL_WEIGHT


# full analysis used by the website
def analyze(cv_text: str, req_text: str) -> dict:
    cv = clean_text(cv_text)
    req = clean_text(req_text)

    cv_skills = find_skill(cv)
    req_skills = find_skill(req)

    skill_part = skill_ratio(cv, req)
    text_part = tfidf_raw(cv, req)
    score = (skill_part * SKILL_WEIGHT + text_part * TFIDF_WEIGHT) * 100

    return {
        "score": round(score, 1),
        "skill_score": round(skill_part * 100, 1),
        "text_score": round(text_part * 100, 1),
        "matched": sorted(cv_skills & req_skills),
        "missing": sorted(req_skills - cv_skills),
        "extra": sorted(cv_skills - req_skills),
        "job_skills_found": len(req_skills),
    }


if __name__ == "__main__":
    cv_raw_text = extract("cv_path.pdf")
    req_raw_text = extract("req_path.pdf")
    result = analyze(cv_raw_text, req_raw_text)
    print(f"Match score: {result['score']:.2f}%")
    print("Matched:", result["matched"])
    print("Missing:", result["missing"])