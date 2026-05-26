import pandas as pd

movies = [
    # Sci-Fi (20 movies)
    (0,  "Inception",          2010, "Sci-Fi",    "A thief who enters the dreams of others to steal secrets is offered a chance to have his record erased if he can plant an idea in a targets mind."),
    (1,  "Interstellar",       2014, "Sci-Fi",    "A team of explorers travel through a wormhole in space to ensure humanitys survival as Earth faces environmental collapse."),
    (2,  "The Matrix",         1999, "Sci-Fi",    "A computer hacker learns reality is a simulation controlled by machines and joins a rebellion to fight back."),
    (3,  "The Martian",        2015, "Sci-Fi",    "An astronaut stranded on Mars must rely on ingenuity to survive while NASA devises a plan to bring him home."),
    (4,  "Gravity",            2013, "Sci-Fi",    "Two astronauts work together to survive after an accident leaves them stranded in outer space."),
    (5,  "Arrival",            2016, "Sci-Fi",    "A linguist works with the military to communicate with alien lifeforms after mysterious spacecraft appear worldwide."),
    (6,  "Ex Machina",         2014, "Sci-Fi",    "A programmer evaluates the human qualities of a highly advanced humanoid AI created by a reclusive tech billionaire."),
    (7,  "Blade Runner 2049",  2017, "Sci-Fi",    "A young blade runner discovers a long buried secret that leads him to track down former blade runner Rick Deckard."),
    (8,  "Moon",               2009, "Sci-Fi",    "An astronaut nearing the end of a three year solo stint on the Moon has a personal encounter that changes everything."),
    (9,  "Annihilation",       2018, "Sci-Fi",    "A biologist joins a secret expedition into a mysterious zone where the laws of nature do not apply."),
    (10, "Contact",            1997, "Sci-Fi",    "A scientist receives a message from extraterrestrials with instructions to build a mysterious machine for travel."),
    (11, "2001 A Space Odyssey",1968,"Sci-Fi",    "Humanity finds a mysterious artifact buried on the moon and sets off on a quest to find its origins with a murderous AI aboard."),
    (12, "Prometheus",         2012, "Sci-Fi",    "A crew of scientists travel to a distant world searching for the origins of humanity but find a terrifying threat instead."),
    (13, "District 9",         2009, "Sci-Fi",    "An extraterrestrial race forced to live in slums on Earth finds a human bureaucrat who becomes their unlikely advocate."),
    (14, "Children of Men",    2006, "Sci-Fi",    "In a dystopian future where humans have become infertile a reluctant man must protect the first pregnant woman in years."),
    (15, "Her",                2013, "Sci-Fi",    "A lonely writer develops an unlikely romantic relationship with an artificially intelligent operating system."),
    (16, "Eternal Sunshine",   2004, "Sci-Fi",    "When a couple undergoes a procedure to erase memories of each other they begin to rediscover their love."),
    (17, "Minority Report",    2002, "Sci-Fi",    "In a future where a special police unit can arrest murderers before they commit crimes an officer is accused of a future murder."),
    (18, "The Truman Show",    1998, "Sci-Fi",    "An insurance salesman discovers his entire life is actually a reality TV show and decides to seek the truth."),
    (19, "Dune",               2021, "Sci-Fi",    "A noble family becomes embroiled in a war for control over a desert planet and its valuable resource that extends life."),

    # Horror (15 movies)
    (20, "Hereditary",         2018, "Horror",    "A grieving family is haunted by disturbing occurrences after the death of their secretive grandmother."),
    (21, "Midsommar",          2019, "Horror",    "A couple travel to Sweden for a midsummer festival that slowly reveals itself to be something deeply sinister."),
    (22, "The Witch",          2015, "Horror",    "A Puritan family in 1630s New England is torn apart by witchcraft black magic and possession after being exiled."),
    (23, "Get Out",            2017, "Horror",    "A young man visits his white girlfriends parents where his growing uneasiness reaches a terrifying boiling point."),
    (24, "It Follows",         2014, "Horror",    "A young woman is followed by an unknown supernatural force after a sexual encounter and must find a way to escape."),
    (25, "A Quiet Place",      2018, "Horror",    "A family struggles to survive in a post-apocalyptic world inhabited by blind monsters with an acute sense of hearing."),
    (26, "The Shining",        1980, "Horror",    "A family heads to an isolated hotel for the winter where a sinister presence influences the father into violence."),
    (27, "Us",                 2019, "Horror",    "A family is attacked by mysterious doppelgangers of themselves while on vacation at a beach house."),
    (28, "Suspiria",           1977, "Horror",    "An American ballet student travels to Germany where she discovers a prestigious dance academy is a front for something sinister."),
    (29, "The Babadook",       2014, "Horror",    "A widowed mother and her son are terrorized by a monster from a mysterious childrens book that appeared in their home."),
    (30, "The Fly",            1986, "Horror",    "A scientist invents a teleportation device but accidentally merges with a fly transforming into a terrifying creature."),
    (31, "The Fly",            1958, "Horror",    "A scientist slowly transforms into a fly after a teleportation experiment goes wrong merging his DNA with an insect."),
    (32, "Rosemarys Baby",     1968, "Horror",    "A young woman suspects her neighbors and husband are part of a satanic cult after she becomes mysteriously pregnant."),
    (33, "The Exorcist",       1973, "Horror",    "A mothers desperate attempts to save her daughter who is possessed by a mysterious satanic entity."),
    (34, "Ari Aster Beau",     2023, "Horror",    "A man plagued by an unexplained phobia must journey through a surreal and nightmarish landscape to find his mother."),

    # Thriller (15 movies)
    (35, "No Country for Old Men", 2007, "Thriller", "Violence erupts after a hunter stumbles upon a drug deal gone wrong and is pursued by a merciless assassin."),
    (36, "Memento",            2000, "Thriller",  "A man with short-term memory loss attempts to track down his wifes murderer using tattoos and notes as clues."),
    (37, "The Prestige",       2006, "Thriller",  "Two stage magicians engage in competitive one upmanship attempting to create the ultimate illusion."),
    (38, "Gone Girl",          2014, "Thriller",  "On the morning of their anniversary a mans wife disappears and the police immediately suspect him of foul play."),
    (39, "Prisoners",          2013, "Thriller",  "When two girls go missing a desperate father takes matters into his own hands while police struggle to find answers."),
    (40, "Zodiac",             2007, "Thriller",  "The true story of the investigation into the Zodiac Killer who terrorized San Francisco with cryptic letters."),
    (41, "Se7en",              1995, "Thriller",  "Two detectives hunt a serial killer who uses the seven deadly sins as his motives in a dark decaying city."),
    (42, "Nightcrawler",       2014, "Thriller",  "A driven young man discovers a career filming nighttime crime scenes for local TV news crossing moral boundaries."),
    (43, "Sicario",            2015, "Thriller",  "An idealistic FBI agent is enlisted by a task force to aid in the escalating war against drugs at the border."),
    (44, "Oldboy",             2003, "Thriller",  "After being imprisoned for 15 years without explanation a man is released and has 5 days to find his captor."),
    (45, "The Silence of the Lambs", 1991, "Thriller", "A young FBI cadet seeks help from an imprisoned cannibal killer to catch another serial killer on the loose."),
    (46, "Parasite",           2019, "Thriller",  "Greed and class discrimination threaten the relationship between the wealthy Park family and the destitute Kim clan."),
    (47, "Knives Out",         2019, "Thriller",  "A detective investigates the death of a famous crime novelist at his estate with all family members as suspects."),
    (48, "The Usual Suspects", 1995, "Thriller",  "A sole survivor tells the story of a ship explosion which may be linked to a mysterious and feared crime lord."),
    (49, "Mulholland Drive",   2001, "Thriller",  "An amnesiac aspiring actress and a famous director become entangled in a dark mystery on the streets of Los Angeles."),

    # Crime (10 movies)
    (50, "The Godfather",      1972, "Crime",     "An aging patriarch of a crime dynasty transfers control to his reluctant son who must protect the family."),
    (51, "Pulp Fiction",       1994, "Crime",     "The lives of two mob hitmen a boxer and a gangsters wife intertwine in darkly comic violent tales."),
    (52, "The Godfather Part II", 1974, "Crime",  "The early life of Vito Corleone contrasts with his sons expansion of the family crime empire."),
    (53, "Goodfellas",         1990, "Crime",     "The story of Henry Hill and his life in the mob covering his relationship with his wife and his career in crime."),
    (54, "The Departed",       2006, "Crime",     "An undercover cop and a mole in the police attempt to identify each other while working for an Irish mob boss."),
    (55, "Heat",               1995, "Crime",     "A group of professional bank robbers start to feel the heat from the LAPD when they start killing witnesses."),
    (56, "Scarface",           1983, "Crime",     "The story of Cuban immigrant Tony Montana who arrives in Miami and becomes a powerful drug lord."),
    (57, "Fargo",              1996, "Crime",     "A car salesman hires two criminals to kidnap his wife but the scheme spirals out of control with deadly consequences."),
    (58, "Reservoir Dogs",     1992, "Crime",     "After a diamond heist goes wrong the surviving criminals gather trying to figure out who among them is a cop."),
    (59, "The Big Lebowski",   1998, "Crime",     "A laid-back slacker is mistaken for a millionaire leading to a series of comic misadventures involving a ransom."),

    # Drama (10 movies)
    (60, "Whiplash",           2014, "Drama",     "A promising drummer enrolls at a cutthroat conservatory where his dreams are threatened by an abusive instructor."),
    (61, "Black Swan",         2010, "Drama",     "A committed dancer wins the lead role in Swan Lake but begins losing her grip on reality as a rival threatens her."),
    (62, "The Shawshank Redemption", 1994, "Drama","Two imprisoned men bond over several years finding solace and eventual redemption through acts of common decency."),
    (63, "Fight Club",         1999, "Drama",     "An insomniac office worker and a soapmaker form an underground fight club that evolves into something dangerous."),
    (64, "Good Will Hunting",  1997, "Drama",     "A janitor at MIT is a natural genius who must work with a therapist to find direction in his troubled life."),
    (65, "The Social Network", 2010, "Drama",     "The story of the founding of Facebook and the legal battles that followed between its creators."),
    (66, "Birdman",            2014, "Drama",     "A faded superhero actor attempts to revive his career by writing and starring in a Broadway production."),
    (67, "Requiem for a Dream",2000, "Drama",     "The drug-induced utopias of four Coney Island people are shattered when their addictions spiral out of control."),
    (68, "12 Years a Slave",   2013, "Drama",     "A free Black man is kidnapped and sold into slavery where he endures years of brutal hardship in the antebellum South."),
    (69, "The Revenant",       2015, "Drama",     "A frontiersman fights for survival after being mauled by a bear and left for dead by members of his own hunting team."),

    # Action (10 movies)
    (70, "The Dark Knight",    2008, "Action",    "When the menacing Joker wreaks havoc on Gotham City Batman must face his greatest psychological and moral test."),
    (71, "Mad Max Fury Road",  2015, "Action",    "In a post-apocalyptic wasteland Max teams with a mysterious woman to flee from a warlord in a high-speed convoy."),
    (72, "John Wick",          2014, "Action",    "An ex-hitman comes out of retirement to track down the gangsters who killed his dog and stole his car."),
    (73, "Die Hard",           1988, "Action",    "A New York cop tries to save his wife and others taken hostage by terrorists in a Los Angeles office building."),
    (74, "The Terminator",     1984, "Action",    "A soldier is sent from 2029 to 1984 to stop a cyborg killing machine sent to assassinate a young woman."),
    (75, "Kill Bill Volume 1", 2003, "Action",    "A former assassin known as The Bride seeks revenge against the gang of assassins who tried to kill her."),
    (76, "Aliens",             1986, "Action",    "A soldier wakes from hypersleep to find colonists on a distant moon have been overrun by terrifying alien creatures."),
    (77, "The Bourne Identity",2002, "Action",    "A man rescued from the ocean with no memory and two bullets in his back must piece together his true identity."),
    (78, "Mission Impossible", 1996, "Action",    "A secret agent is framed for the deaths of his entire team and must find the actual culprit to clear his name."),
    (79, "Avengers Endgame",   2019, "Action",    "After Thanos destroys half of all life the remaining Avengers must do whatever it takes to reverse his actions."),

    # Animation (8 movies)
    (80, "Spirited Away",      2001, "Animation", "During her familys move a 10-year-old girl wanders into a world ruled by gods witches and spirits."),
    (81, "Princess Mononoke",  1997, "Animation", "On a journey to find a cure for a curse Ashitaka finds himself in a war between forest gods and humans."),
    (82, "WALL-E",             2008, "Animation", "A small waste-collecting robot discovers love while inadvertently igniting a quest for humanitys return to Earth."),
    (83, "Up",                 2009, "Animation", "A 78-year-old man travels to South America by tying balloons to his house accompanied by an earnest young boy."),
    (84, "Toy Story",          1995, "Animation", "A cowboy doll is threatened when a new spaceman figure supplants him as top toy in a boys room."),
    (85, "The Lion King",      1994, "Animation", "A young lion prince flees his kingdom after the murder of his father and learns the true meaning of responsibility."),
    (86, "Coco",               2017, "Animation", "Aspiring musician Miguel enters the Land of the Dead to find his great-great-grandfather and lift a family curse."),
    (87, "Spider-Man Spider-Verse", 2018, "Animation", "Teen Miles Morales becomes Spider-Man and teams up with versions of the hero from parallel universes."),

    # Romance (6 movies)
    (88, "La La Land",         2016, "Romance",   "A pianist and an actress fall in love in Los Angeles while attempting to reconcile their aspirations for the future."),
    (89, "Before Sunrise",     1995, "Romance",   "A young American man meets a French woman on a train and they spend one night together in Vienna before parting."),
    (90, "The Notebook",       2004, "Romance",   "A poor but passionate young man falls in love with a rich young woman and they share a summer romance they never forget."),
    (91, "Crazy Stupid Love",  2011, "Romance",   "A middle-aged man is taught how to be a player by a younger man but the lessons backfire in unexpected ways."),
    (92, "Amélie",             2001, "Romance",   "A shy Parisian woman decides to change the lives of those around her for the better while struggling with her own love life."),
    (93, "500 Days of Summer", 2009, "Romance",   "An offbeat romantic comedy about a woman who doesnt believe true love exists and the man who falls for her anyway."),

    # Mystery (6 movies)
    (94, "Rear Window",        1954, "Mystery",   "A photographer with a broken leg spies on neighbors from his window and believes he witnessed a murder."),
    (95, "Chinatown",          1974, "Mystery",   "A private detective hired to investigate adultery uncovers a sinister conspiracy involving the citys water supply."),
    (96, "The Usual Suspects", 1995, "Mystery",   "A sole survivor tells the story of a robbery which may be linked to a mysterious and feared crime lord named Keyser Soze."),
    (97, "Vertigo",            1958, "Mystery",   "A retired detective with a fear of heights is hired to follow a woman and becomes dangerously obsessed with her."),
    (98, "Rear Window",        1954, "Mystery",   "A laid-up photographer spies on his neighbors through his window believing one of them has committed murder."),
    (99, "Knives Out 2",       2022, "Mystery",   "Detective Benoit Blanc travels to Greece to investigate the mysterious death of a tech billionaire on his private island."),

    # Comedy (4 movies)
    (100, "The Grand Budapest Hotel", 2014, "Comedy", "A writer encounters the owner of an aging European hotel and learns of his adventures with the legendary concierge."),
    (101, "Superbad",          2007, "Comedy",    "Two co-dependent high school seniors plot to score alcohol for a party in an attempt to impress girls before graduation."),
    (102, "The Hangover",      2009, "Comedy",    "Three friends wake up after a bachelor party in Las Vegas with no memory and must find the missing groom."),
    (103, "Knives Out",        2019, "Comedy",    "A witty detective investigates the death of a famous crime novelist at his estate in a darkly comic whodunit."),
]

df = pd.DataFrame(movies, columns=["id","title","year","genre","description"])
df.to_csv("data/movies.csv", index=False)
print(f"Generated {len(df)} movies")
print(df["genre"].value_counts())