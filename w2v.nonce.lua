--[[
Class for word2vec with skipgram and negative sampling
--]]

require("sys")
require("nn")
require 'util.backgroundSpace'
require 'util.backgroundVocab'
local Word2Vec = torch.class("Word2Vec")

function Word2Vec:__init(config)
    self.tensortype = torch.getdefaulttensortype()
    self.gpu = config.gpu -- 1 if train on gpu, otherwise cpu
    self.stream = config.stream -- 1 if stream from hard drive, 0 otherwise
    self.neg_samples = config.neg_samples
    self.dim = config.dim
    self.criterion = nn.BCECriterion() -- logistic loss
    --self.criterion = nn.MSECriterion()
    self.word = torch.IntTensor(1) 
    self.contexts = torch.IntTensor(1+self.neg_samples) 
    self.labels = torch.zeros(1+self.neg_samples); self.labels[1] = 1 -- first label is always pos sample
    self.window = config.window 
    self.lr = config.lr 
    self.min_lr = config.min_lr
    self.alpha = config.alpha
    self.table_size = config.table_size 
    --self.background_freqs = config.background_freqs
    self.vocab = {}
    self.index2word = {}
    self.word2index = {}
    self.index2prob = {}
    self.total_count = 0
    self.batch_size = config.batch_size
    self.loss_toprint = 0.0
    self.subsampl = config.subsampl
    self.epochs = config.epochs
end

function Word2Vec:reinit(config)
    -- Make sure the nonce is back to a random representation
     for i=1, self.dim do
         self.word_vecs.weight[self.word2index["___"]][i] = torch.normal(0, 0.9)
     end
    self.gpu = config.gpu -- 1 if train on gpu, otherwise cpu
    self.stream = config.stream -- 1 if stream from hard drive, 0 otherwise
    self.neg_samples = config.neg_samples
    self.dim = config.dim
    self.criterion = nn.BCECriterion() -- logistic loss
    --self.criterion = nn.MSECriterion()
    self.window = config.window 
    self.lr = config.lr 
    self.min_lr = config.min_lr
    self.alpha = config.alpha
    self.table_size = config.table_size 
    --self.background_freqs = config.background_freqs
    self.vocab = {}
    self.index2word = {}
    self.word2index = {}
    self.index2prob = {}
    self.total_count = 0
    self.batch_size = config.batch_size
    self.loss_toprint = 0.0
    self.subsampl = config.subsampl
    self.epochs = config.epochs
end

------------------------------------------------
-- Initialise model
------------------------------------------------

function Word2Vec:initialise_model(background)
    self.original_lr = self.lr
    self.word_vecs = backgroundSpace(self.word2index, self.dim, background) -- word embeddings from background
    self.context_vecs = backgroundSpace(self.word2index, self.dim, background) -- word embeddings from background
    self.original_context_vecs = backgroundSpace(self.word2index, self.dim, background) -- make a copy because we don't want the context vectors to change
    
    self.batchlabels = torch.zeros(self.batch_size,self.neg_samples+1)
    self.batchlabels[{{},1}]:fill(1)
    
    self.w2v = nn.Sequential()
    self.w2v:add(nn.ParallelTable())
    self.w2v.modules[1]:add(self.context_vecs)
    self.w2v.modules[1]:add(self.word_vecs)
    self.w2v:add(nn.MM(false, true)) -- dot prod and sigmoid to get probabilities
    self.w2v:add(nn.Sigmoid())
    self.decay = (self.min_lr-self.lr)/(self.total_count*self.window)
    --print(self.word_vecs.weight)
    --print(self.context_vecs.weight)
    --print(self.context_vecs)
end 


-----------------------------------------------------------------------
-- Build vocab frequency, word2index, and index2word from input file
-----------------------------------------------------------------------

function Word2Vec:build_vocab(corpus,background)
    --print("Building vocabulary...")
    local start = sys.clock()
    local f = io.open(corpus, "r")
    local n = 1
    if background ~= "none" then 
	self.vocab = backgroundVocab(background).vocabulary
    end
    for line in f:lines() do
        for _, word in ipairs(self:split(line)) do
	    self.total_count = self.total_count + 1
	    if self.vocab[word:lower()] == nil then
	        self.vocab[word:lower()] = 1	 
	    	--print(word,self.vocab[word:lower()])
            else
	        self.vocab[word:lower()] = self.vocab[word:lower()] + 1
	    end
	    --print(word,self.vocab[word:lower()])
        end
        n = n + 1
    end
    f:close()
  
    for word, count in pairs(self.vocab) do
        self.index2word[#self.index2word+1] = word
        self.word2index[word] = #self.index2word	    
    end

    self.vocab_size = #self.index2word
    --print(string.format("%d words and %d sentences processed in %.2f seconds.", self.total_count, n, sys.clock() - start))
    --print(string.format("Vocab size: %d", self.vocab_size))
    
    --print(self.index2word)
    self:initialise_model(background)
end



--------------------------------------------------------------------------------
-- Build a table of unigram probabilities from which to obtain negative samples
--------------------------------------------------------------------------------

function Word2Vec:build_table_current()
    local start = sys.clock()
    local total_count_pow = 0
    --print("Building a table of unigram probabilities... ")
    for _, count in pairs(self.vocab) do
    	total_count_pow = total_count_pow + count^self.alpha
    end   
    self.table = torch.IntTensor(self.table_size)						-- Create table of a particular size (the bigger the better)
    local word_index = 1									-- Set word index to 1
    local word_prob = self.vocab[self.index2word[word_index]]^self.alpha / total_count_pow	-- Calculate probability of word_index (i.e. prob of first word in vocab)
    for idx = 1, self.table_size do								-- For each position in the table...
        self.table[idx] = word_index								-- we repeat words as many times as it takes to have more frequent words more likely to sample
        if idx / self.table_size > word_prob then						-- If the position in the table (as a ratio of the table_size) is greater than the prob of current word...
            word_index = word_index + 1								-- it's time to switch to the next word
            if self.vocab[self.index2word[word_index]] ~= nil then
	    	word_prob = word_prob + self.vocab[self.index2word[word_index]]^self.alpha / total_count_pow	-- we add the probabilities of the previous words and the current one
	    end
        end
        if word_index > self.vocab_size then
            word_index = word_index - 1
        end
    end
    --print(string.format("Done in %.2f seconds.", sys.clock() - start))
end



-------------------------------
-- Sample negative contexts
-------------------------------
function Word2Vec:sample_contexts(context,word)
    --print("Sampling negative contexts...")
    local start=sys.clock()
    self.contexts[1] = context
    local i = 0
    while i < self.neg_samples do
        rand = torch.random(self.table_size)
        start=sys.clock()
        neg_context = self.table[rand]
	if context ~= neg_context then
	    self.contexts[i+2] = neg_context
	    i = i + 1
	end
    end
end


--------------------------------
-- Train on word context pairs
--------------------------------

function Word2Vec:train_pair(word, contexts)
    --print("Training pair...")
    local start = sys.clock()
    local batchlabels = self.batchlabels
    if self.gpu == 1 then
	    word = word:clone():cuda()
	    contexts = contexts:clone():cuda()
	    batchlabels = batchlabels:clone():cuda()
    end
    --print(word,contexts,batchlabels)
    self.w2v:zeroGradParameters()
    local p = self.w2v:forward({contexts,word})						-- do forward pass through MM and sigmoid, get output 
    local loss = self.criterion:forward(p, batchlabels)						-- compare with actual labels (one-hot vector, with first element set to one)
    self.loss_toprint=self.loss_toprint+loss
    --self.loss_toprint=loss
    local dl_dp = self.criterion:backward(p, batchlabels)
    self.w2v:backward({contexts, word}, dl_dp)
    self.w2v:updateParameters(self.lr)
    --print(self.word_vecs.weight)
    --print(self.context_vecs.weight)
end



----------------------------------------------------------------------------------------
-- Train on sentences that are streamed from the hard drive
----------------------------------------------------------------------------------------

function Word2Vec:train_stream(corpus)
    --print("Training (streaming)...")
    local start = sys.clock()
    local c = 0				-- word count
    local total_c = 0			-- total word count
    local snippet = ""

    local f = io.input(corpus)   -- open input file
    for line in f:lines() do
	start = sys.clock()
	sentence = self:split(line)
	len_sentence=#sentence
	total_c=total_c+#sentence

	-- Create subsampled line. We only keep the words that are in the vocab.	
	local subsampled_line=""
	--print(sentence)
	for i, word in ipairs(sentence) do
	    word=word:lower()
	    word_idx = self.word2index[word]
	    freq=self.vocab[self.index2word[word_idx]]
	    if freq ~= nil then -- word exists in vocab
		p_keep = (torch.sqrt(freq / (self.subsampl * self.total_count)) + 1) * (self.subsampl * self.total_count) / freq
		random=torch.random()/(2^32)
		--print(freq,self.total_count)
		--print(p_keep,random)
		if p_keep > random then
	            subsampled_line = subsampled_line..word
	            subsampled_line = subsampled_line.." "
		end
	    end
	end

	local batch_word = torch.zeros(self.batch_size,1)
	local batch_contexts = torch.zeros(self.batch_size,self.neg_samples+1)

	--local reduced_window = torch.random(self.window) -- pick random window size
	local reduced_window = self.window -- If only seeing the sentence once, keep whole window
	--print(#sentence,reduced_window)
	count = 1
	print("SUB LINE:",subsampled_line)
	sentence = self:split(subsampled_line)
	for i,word in ipairs(sentence) do
		--print(word)
		--print(self.word2index[word])
		--if word == "___" then				-- only train the gap
		word_idx = self.word2index[word]
		if word_idx ~= nil then
			self.word[1] = word_idx -- update current word
			for j = i - reduced_window, i + reduced_window do -- loop through contexts
			    local context = sentence[j]
			    if context ~= nil and j ~= i then -- possible context
				--print(context)
				--print(i,j)
				context_idx = self.word2index[context]
				if context_idx ~= nil then -- valid context
					self:sample_contexts(context_idx,word_idx) -- update pos/neg contexts
					if count > self.batch_size
					then
						self.context_vecs=self.original_context_vecs			-- reset contexts because we don't want them to change.
						self:train_pair(batch_word, batch_contexts)
						self.lr = math.max(self.min_lr, self.lr + self.decay)
						count = 1	
						batch_word = torch.zeros(self.batch_size,1)			--reset the batches
						batch_contexts = torch.zeros(self.batch_size,self.neg_samples+1)
						--print(word,context)
						if self.index2word[self.word] ~= nil
						then
							batch_word[count]=self.word
							batch_contexts[count]=self.contexts
							count = count + 1
						end
					else
						--print(word,context)
						if self.word2index[word] ~= nil
						then
							batch_word[count]=self.word2index[word]
							batch_contexts[count]=self.contexts
							count = count + 1
							--print(count)
						end
					end
					--WARNING: if using batch_size > 1, the end of sentence might not be read.
							
				c = c + 1
				end
			end
		    --end
		end
	    end
	end
	--print(string.format("%d word pairs trained in %.2f seconds. Learning rate: %.4f", c, sys.clock() - start, self.lr))
	--print("Total count:",total_c)
	self.loss_toprint=self.loss_toprint/c
	--print(string.format("Loss: %.7f",self.loss_toprint))
	--self:print_sim_words({"___"},10)
	self.loss_toprint=0.0
	c=0
    end
end


--------------------------------------------
-- Train the model using config parameters
--------------------------------------------

function Word2Vec:train_model(corpus)
    if self.gpu==1 then
        self:cuda()
    end
    if self.stream==1 then
        self:train_stream(corpus)
    else
        self:preload_data(corpus)
	self:train_mem()
    end
end






--------------------------------------------------------------------------------------
-- Helper functions (similarity, printing, etc)
--------------------------------------------------------------------------------------

-------------------------
-- move to cuda
-------------------------
function Word2Vec:cuda()
    require("cunn")
    require("cutorch")
    cutorch.setDevice(1)
    self.criterion:cuda()
    self.w2v:cuda()
end

----------------------
-- Split on separator
----------------------

function Word2Vec:split(input, sep)
    if sep == nil then
        sep = "%s"
    end
    local t = {}; local i = 1
    for str in string.gmatch(input, "([^"..sep.."]+)") do
        t[i] = str; i = i + 1
    end
    return t
end
----------------------------------------------
-- Dump function to write out vector elements
----------------------------------------------

local dump = function(vec)
    vec = vec:view(vec:nElement())
    local t = {}
    for i=1,vec:nElement() do
        t[#t+1] = string.format('%.4f', vec[i])
    end
    return table.concat(t, '\t')
end

--------------------------
-- Row-normalize a matrix
--------------------------

function Word2Vec:normalize(m)
    m_norm = torch.zeros(m:size())
    for i = 1, m:size(1) do
    	m_norm[i] = m[i] / torch.norm(m[i])
    end
    return m_norm
end
----------------------------
-- Normalize between 0 and 1
----------------------------

function Word2Vec:normalise_positive(m)
    m_norm = (m-torch.min(m))/(torch.max(m)-torch.min(m))
    return m_norm
end

--------------------------------------------------
-- Print semantic space in DISSECT .dm format
--------------------------------------------------

function Word2Vec:print_semantic_space()
    self.word_vecs_norm = self:normalize(self.word_vecs.weight:double())
    for word,_ in pairs(self.vocab) do
	vec=self.word_vecs_norm[self.word2index[word]]
	vec:resize(vec:size(1),1)				--can only transpose vector with dim 2, so resize
	vec=vec:t()
	io.write(word," ",dump(vec)," ")
    end
end

------------------------------------------------------------------------------
-- Return the k-nearest words to a word or a vector based on cosine similarity
------------------------------------------------------------------------------

function Word2Vec:get_sim_words(w, k)
    self.word_vecs_norm = self:normalize(self.word_vecs.weight:double())
    if type(w) == "string" then
        if self.word2index[w] == nil then
	   print("'"..w.."' does not exist in vocabulary.")
	   return nil
	else
            w = self.word_vecs_norm[self.word2index[w]]
	end
    end
    local sim = torch.mv(self.word_vecs_norm, w)
    sim, idx = torch.sort(-sim)
    local r = {}
    for i = 1, k do
        r[i] = {self.index2word[idx[i]], -sim[i]}
    end
    return r
end

------------------------------------------------------------------------------
-- Return the rank of a word in list of nearest neighbours
------------------------------------------------------------------------------

function Word2Vec:get_rank_sim(w1, w2)
    self.word_vecs_norm = self:normalize(self.word_vecs.weight:double())
    if type(w1) == "string" and type(w2) == "string" then
        if self.word2index[w1] == nil or self.word2index[w2] == nil then
	   print(w1.." or "..w2.." does not exist in vocabulary.")
	   return nil
	else
            w1 = self.word_vecs_norm[self.word2index[w1]]
	end
    end
    local sim = torch.mv(self.word_vecs_norm, w1)
    sim, idx = torch.sort(-sim)
    r = -1
    for i = 1, 1000 do
        if w2 == self.index2word[idx[i]] then
	    r = i
	    break
	end
    end
   return r
end

------------------------------------------------------------------------------
-- Return the similarity of two words
------------------------------------------------------------------------------
function Word2Vec:cosine(w1, w2)
    self.word_vecs_norm = self:normalize(self.word_vecs.weight:double())
    if type(w1) == "string" then
        if self.word2index[w1] == nil or self.word2index[w2] == nil then
	   print(w1.." or "..w2.." does not exist in vocabulary.")
	   return nil
	else
            w1 = self.word_vecs_norm[self.word2index[w1]]
            w2 = self.word_vecs_norm[self.word2index[w2]]
	end
    end
    w1 = torch.Tensor(1,w1:size()[1]):copy(w1)
    local sim = torch.mv(w1,w2)
--    wn1 = self.word_vecs_norm[self.word2index[w1]]
--    wn2 = self.word_vecs_norm[self.word2index[w2]]
--    local sim = (wn1:dot(wn2) / (math.sqrt(wn1:dot(wn1)) * math.sqrt(wn2:dot(wn2))))
    return sim[1]
end

-------------------------------
-- Print k neighbours
-------------------------------

function Word2Vec:print_sim_words(words, k)
    for i = 1, #words do
        sim_words=""
    	r = self:get_sim_words(words[i], k)
	if r ~= nil then
   	    --print("-------"..words[i].."-------")
	    sim_words=sim_words..words[i]..": "
	    for j = 1, k do
	        --print(string.format("%s, %.4f", r[j][1], r[j][2]))
		sim_words=sim_words..string.format("%s, (%.4f)",r[j][1],r[j][2]).." "
	    end
	    print(sim_words)
	end
    end
end


