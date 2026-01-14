def play():
    with open("plays.txt","r") as f:
        plays = int(f.read())
        plays += 1
    with open("plays.txt","w") as f:
         f.write(str(plays))
    return str(plays)
