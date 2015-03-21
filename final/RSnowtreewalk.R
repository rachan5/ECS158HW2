SNOW <- function(x,size,root,type=c("descendants")){
    ans <- rep(0,size)
    mystart <- (myid-1)*length(x)+1
    myend <- myid*length(x)
    
    type <- match.arg(type)
    if (type == "descendants"){
        v1 <- descendant
        v2 <- ancestor
        #initalization
        temp <- v1[mystart:myend]
        
        #second and beyond iteration
        for (j in 1:size){ 
            if (node %in% temp){
                setthese <- which(temp == node) + mystart-1
                ans[setthese] <- 1
            }
            blah <- rep(-1,length(temp))
            for (i in (1:length(temp))){
                matched_pos <- which(v1 == temp[i])
                if (length(matched_pos) != 0){
                    blah[which(temp == temp[i])] <- matched_pos
                }
                else{#matched_pos == 0
                    ## R is 1 INDEXED!
                    if (type == "descendants"){
                        blah[i] <- 1
                    }
                }
            }#for i 
            #"go to your parents set"
            difference <- length(temp) - length(v2[blah])
            temp <- v2[blah]
            if (difference > 0){
                temp <- c(rep(0,difference),temp)
            }
            if (node %in% temp){
                setthese <- which(temp == node) + mystart-1
                ans[setthese] <- 1
            }
        }#j loop
    }#new endif for type==descendants
    return(ans)
}# end SNOW

setmyid <- function(i){
    myid <<- i
}

## get descendants with RSnow
RSnowdescendants <- function (phy, node, type=c("tips","children","all"),cls) {
    type <- match.arg(type)

    ## look up nodes, warning about and excluding invalid nodes
    oNode <- node
    node <- getNode(phy, node, missing="warn")
    isValid <- !is.na(node)
    node <- as.integer(node[isValid])

    if (type == "children") {
        res <- lapply(node, function(x) children(phy, x))
        ## if just a single node, return as a single vector
        if (length(res)==1) res <- res[[1]]
    } else {
        ## edge matrix must be in preorder for the C function!
        #if (phy@order=="preorder") {
            edge <- phy@edge
        #} else {
        #    edge <- reorder(phy, order="postorder")@edge
        #}
        ## extract edge columns
        ancestor <- as.integer(edge[, 1])
        descendant <- as.integer(edge[, 2])
        
        ## return indicator matrix of ALL descendants (including self)
        #isDes <- .Call("descendants", node, ancestor, descendant)
        clusterExport(cls,c("node", "ancestor", "descendant","setmyid","SNOW"),envir=environment())
        dexgrps <- splitIndices(length(ancestor),length(cls))
        rootdex <- which(phy@edge[,1] == 0)
        clusterApply(cls,1:length(cls),setmyid)
        newisDes <- clusterApply(cls,dexgrps,SNOW,length(ancestor),rootdex,"descendants")
        isDes <- (matrix(Reduce('+',newisDes),nrow=length(ancestor),ncol=1))
        storage.mode(isDes) <- "logical"

        ## for internal nodes only, drop self (not sure why this rule?)
        int.node <- intersect(node, nodeId(phy, "internal"))
        isDes[cbind(match(int.node, descendant),
            match(int.node, node))] <- FALSE
        
        ## if only tips desired, drop internal nodes
        if (type=="tips") {
            isDes[descendant %in% nodeId(phy, "internal"),] <- FALSE
        }
        ## res <- lapply(seq_along(node), function(n) getNode(phy,
        ##     descendant[isDes[,n]]))
        res <- getNode(phy, descendant[isDes[, seq_along(node)]])
    }
    ## names(res) <- as.character(oNode[isValid])

    res
}

###############
# shortestPath
###############

RSnowshortestPath <- function(phy, node1, node2,cls){

    ## conversion from phylo, phylo4 and phylo4d
    if (class(phy) == "phylo4d") {
        x <- extractTree(phy)
    }
    else if (class(phy) != "phylo4"){
        x <- as(phy, "phylo4")
    }
    ## some checks
    t1 <- getNode(x, node1)
    t2 <- getNode(x, node2)
    if(any(is.na(c(t1,t2)))) stop("wrong node specified")
    if(t1==t2) return(NULL)

    ## main computations
    comAnc <- MRCA(x, t1, t2) # common ancestor
    desComAnc <- RSnowdescendants(x, comAnc, type="all",cls)
    ancT1 <- ancestors(x, t1, type="all")
    path1 <- intersect(desComAnc, ancT1) # path: common anc -> t1

    ancT2 <- ancestors(x, t2, type="all")
    path2 <- intersect(desComAnc, ancT2) # path: common anc -> t2

    res <- union(path1, path2) # union of the path
    ## add the common ancestor if it differs from t1 or t2
    if(!comAnc %in% c(t1,t2)){
        res <- c(comAnc,res)
    }

    res <- getNode(x, res)

    return(res)
} # end shortestPath
