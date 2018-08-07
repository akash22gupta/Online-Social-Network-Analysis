# Implemented community detection and link prediction algorithms using Facebook "like" data.

# The file `edges.txt.gz` indicates like relationships between facebook users. This was collected using snowball sampling: beginning with the user "Bill Gates", crawled all the people he "likes", then, for each newly discovered user, crawled all the people they liked.

# Clustered the resulting graph into communities, as well as recommended friends for Bill Gates.

