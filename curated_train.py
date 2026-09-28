import train_real_faces as m

# Curated Commons categories whose subject is the named public figure.
# This avoids broad text search and avoids crawling unrelated surname/name categories.
m.PEOPLE = [
    (
        "taylor_swift",
        "Taylor Swift",
        [
            "Taylor Swift in 2007",
            "Taylor Swift in 2019",
            "Taylor Swift in 2023",
            "Taylor Swift in 2024",
        ],
    ),
    (
        "zendaya",
        "Zendaya",
        [
            "Zendaya in 2019",
            "Zendaya in 2024",
            "Zendaya in 2026",
        ],
    ),
    (
        "tom_holland",
        "Tom Holland",
        [
            "Tom Holland (actor) at the 2016 Comic-Con International",
            "Tom Holland (actor) in 2018",
            "Tom Holland (actor) in 2026",
        ],
    ),
]

if __name__ == "__main__":
    m.collect()
    m.train()
