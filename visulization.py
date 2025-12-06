from loadfile_datapreproccessing import df
import matplotlib.pyplot as plt
import seaborn as sns

#display rating

plt.hist(df['Rating'])
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

#display relation between rating and year

plt.scatter(df['Year'], df['Rating'])
plt.xlabel("Year")
plt.ylabel("Rating")
plt.show()

#display relation between rating and Votes
plt.scatter(df['Votes'], df['Rating'])
plt.xscale("log")
plt.xlabel("Votes (log scale)")
plt.ylabel("Rating")
plt.show()

#display rating intervals with seaborn 

sns.histplot(df['Rating'], bins=10, kde=True, color='skyblue')
plt.xlabel("Rating")     
plt.ylabel("Count")      
plt.title("Distribution of Movie Ratings") 
plt.show()

#display relation between rating and ReviewCount
plt.scatter(df['ReviewCount'], df['Rating'])
plt.xscale("log")
plt.xlabel("Votes (log scale)")
plt.ylabel("Rating")
plt.show()
