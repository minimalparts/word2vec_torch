--[[
Class for word2vec with skipgram and negative sampling
--]]

require("sys")
require("nn")
require 'util.backgroundSpace'

--local MinibatchLoader = require 'util.MinibatchLoader'
local Word2Vec = torch.class("Word2Vec")

function Word2Vec:__init(config)
    self.tensortype = torch.getdefaulttensortype()
    self.gpu = config.gpu -- 1 if train on gpu, otherwise cpu
    self.stream = config.stream -- 1 if stream from hard drive, 0 otherwise
    self.neg_samples = config.neg_samples
    self.minfreq = config.minfreq
    self.dim = config.dim
    self.criterion = nn.BCECriterion() -- logistic loss
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
end

-- move to cuda
function Word2Vec:cuda()
    require("cunn")
    require("cutorch")
    cutorch.setDevice(1)
    self.word = self.word:cuda()
    self.contexts = self.contexts:cuda()
    self.labels = self.labels:cuda()
    self.criterion:cuda()
    self.w2v:cuda()
end

function Word2Vec:initialise_model(background)
    if background=="none" then
        -- initialize word/context embeddings now that vocab size is known
        self.word_vecs = nn.LookupTable(self.vocab_size, self.dim, self.gpu) -- word embeddings
        self.context_vecs = nn.LookupTable(self.vocab_size, self.dim, self.gpu) -- context embeddings
        self.word_vecs:reset(0.25); self.context_vecs:reset(0.25) -- rescale N(0,1)
    else
       self.word_vecs = backgroundSpace(self.word2index, self.dim, background) -- word embeddings from background
       self.context_vecs = backgroundSpace(self.word2index, self.dim, background) -- word embeddings from background
    end
    self.w2v = nn.Sequential()
    self.w2v:add(nn.ParallelTable())
    self.w2v.modules[1]:add(self.context_vecs)
    self.w2v.modules[1]:add(self.word_vecs)
    self.w2v:add(nn.MM(false, true)) -- dot prod and sigmoid to get probabilities
    --self.w2v:add(nn.Sigmoid())
    self.w2v:add(nn.SoftMax())
    self.decay = (self.min_lr-self.lr)/(self.total_count*self.window)
end 

-- Build vocab frequency, word2index, and index2word from input file
function Word2Vec:build_vocab(corpus,background)
    print("Building vocabulary...")
    local start = sys.clock()
    local f = io.open(corpus, "r")
    local n = 1
    for line in f:lines() do
        for _, word in ipairs(self:split(line)) do
	    self.total_count = self.total_count + 1
	    if self.vocab[word:lower()] == nil then
	        self.vocab[word:lower()] = 1	 
            else
	        self.vocab[word:lower()] = self.vocab[word:lower()] + 1
	    end
	    --print(word,self.vocab[word])
        end
        n = n + 1
    end
    f:close()
   
    if background ~= "none" then 
	    -- Add vocab from the background space
	    for line in io.lines(background) do
		local parts = split(line, " ")
		local word = parts[1]
		if self.vocab[word:lower()] == nil then
		    self.vocab[word:lower()] = 1	 
		else
		    self.vocab[word:lower()] = self.vocab[word:lower()] + 1
		end
	    end
    end

    -- Delete words that do not meet the minfreq threshold and create word indices
    for word, count in pairs(self.vocab) do
    	if count >= self.minfreq then
     	    self.index2word[#self.index2word+1] = word
            self.word2index[word] = #self.index2word	    
    	else
	    self.vocab[word] = nil
        end
    end
    --torch.save(background..".t7", self.vocab)
    self.vocab_size = #self.index2word
    print(string.format("%d words and %d sentences processed in %.2f seconds.", self.total_count, n, sys.clock() - start))
    print(string.format("Vocab size after eliminating words occuring less than %d times: %d", self.minfreq, self.vocab_size))
    
    self:initialise_model(background)
end


-- Build a table of unigram probabilities from which to obtain negative samples (from current corpus, not background!!)

-- Build a table of unigram frequencies from which to obtain negative samples
function Word2Vec:build_table_current()
    local start = sys.clock()
    local total_count_pow = 0
    print("Building a table of unigram probabilities... ")
    for _, count in pairs(self.vocab) do
    	total_count_pow = total_count_pow + count^self.alpha
    end   
    self.table = torch.IntTensor(self.table_size)
    local word_index = 1
    local word_prob = self.vocab[self.index2word[word_index]]^self.alpha / total_count_pow
    for idx = 1, self.table_size do
        self.table[idx] = word_index									--We repeat words as many times as it takes to have more frequent words more likely to sample
        if idx / self.table_size > word_prob then
            word_index = word_index + 1
            if self.vocab[self.index2word[word_index]] ~= nil then --word is in vocab (AH fix)
	    	word_prob = word_prob + self.vocab[self.index2word[word_index]]^self.alpha / total_count_pow
	    end
        end
        if word_index > self.vocab_size then
            word_index = word_index - 1
        end
    end
    print(string.format("Done in %.2f seconds.", sys.clock() - start))
end

-- Train on word context pairs
function Word2Vec:train_pair(word, contexts)
    --local start = sys.clock()
    local batchlabels = torch.zeros(self.batch_size,self.neg_samples+1)
    if self.gpu == 1 then
	    batchlabels = batchlabels:cuda()
	    for i=1,self.batch_size do
		batchlabels[i] = self.labels
	    end
    end
    local p = self.w2v:forward({contexts,word})						-- do forward pass through MM and sigmoid, get output 
    local loss = self.criterion:forward(p, batchlabels)						-- compare with actual labels (one-hot vector, with first element set to one)
    local dl_dp = self.criterion:backward(p, batchlabels)
    self.w2v:zeroGradParameters()
    self.w2v:backward({contexts, word}, dl_dp)
    self.w2v:updateParameters(self.lr)
    --print(string.format("Trained pair in %.2f seconds.", sys.clock() - start))
end


-- Sample negative contexts
function Word2Vec:sample_contexts(context)
    self.contexts[1] = context
    local i = 0
    while i < self.neg_samples do
        neg_context = self.table[torch.random(self.table_size)]
	if context ~= neg_context then
	    self.contexts[i+2] = neg_context
	    i = i + 1
	end
    end
end

-- split on separator
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

-- pre-load data as a torch tensor instead of streaming it. this requires a lot of memory, 
-- so if the corpus is huge you should partition into smaller sets
function Word2Vec:preload_data(corpus)
    print("Preloading training corpus into tensors (Warning: this takes a lot of memory)")
    local start = sys.clock()
    local c = 0
    local keep = 1
    f = io.open(corpus, "r")
    self.train_words = {}; self.train_contexts = {}
    for line in f:lines() do										-- loop through each line in corpus
        sentence = self:split(line)									-- split line into words
	local subsampled_line= ""
        for i, word in ipairs(sentence) do								-- for each word...
	    word=word:lower()										-- convert to lower case...
	    word_idx = self.word2index[word]								-- get the index of the word...
	    freq=self.vocab[self.index2word[word_idx]]								-- get frequency of the word in the corpus...
	    if freq ~= nil then
		    subsampl=(torch.sqrt(freq/(10^-4*self.total_count))+1)*(10^-4*self.total_count)/freq									-- subsampling formula
		    --print("WORD SUB",word,freq,subsampl,keep)
		    if subsampl > 0.2 then
			subsampled_line = subsampled_line..word
			subsampled_line = subsampled_line.." "
		    end
	    end
        end
	--print(subsampled_line)
	sentence = self:split(subsampled_line)
    	local reduced_window = torch.random(self.window) -- pick random window size
	for i,word in ipairs(sentence) do
		word_idx = self.word2index[word]								-- get the index of the word...
		self.word[1] = word_idx -- update current word						-- this is confusing... self.word is not word... should be called 'target'
		for j = i - reduced_window, i + reduced_window do -- loop through contexts
			local context = sentence[j]								-- grab surface form of this context
			if context ~= nil and j ~= i then -- possible context				-- if the context is not nil, and not the target word
				context_idx = self.word2index[context]						-- get the context's id
				if context_idx ~= nil then -- valid context					-- check id is not nil
				    c = c + 1									-- increment overall word-context count
				    self:sample_contexts(context_idx) -- update pos/neg contexts		-- get negative contexts (this has the effect of filling self.contexts with neg context ids)
				    if self.gpu==1 then
					self.train_words[c] = self.word:clone():cuda()				-- add ids to overall training data
					self.train_contexts[c] = self.contexts:clone():cuda()
				    else
					self.train_words[c] = self.word:clone()
					self.train_contexts[c] = self.contexts:clone()
				    end
				end
		       end	      
		end
	end
    end
    print(string.format("%d word-contexts processed in %.2f seconds", c, sys.clock() - start))
end

-- train from memory. this is needed to speed up GPU training
function Word2Vec:train_mem()
    local start = sys.clock()
    count = 1
    local batch_word = torch.zeros(self.batch_size,1)
    local batch_contexts = torch.zeros(self.batch_size,self.neg_samples+1)
    if self.gpu == 1 then
	batch_word = batch_word:clone():cuda()
	batch_contexts = batch_contexts:clone():cuda()
    end
    for i = 1, #self.train_words do
	if count > self.batch_size
	then
		--print(self.train_words[i], self.train_contexts[i])
		-- one tensor with one element (the target idx), one tensor with neg_samples + 1 elements: the actual context id and the neg ones
		--self:train_pair(self.train_words[i], self.train_contexts[i])
		--for j = 1, self.batch_size do
		--	print(self.index2word[batch_word[j]],self.index2word[batch_contexts[j][1]],self.index2word[batch_contexts[j][2]])
		--end
		--print(batch_word,batch_contexts)
		self:train_pair(batch_word, batch_contexts)
		self.lr = math.max(self.min_lr, self.lr + self.decay)
		count = 1	
		for k=1,self.batch_size do
			batch_word[k]=0.0
			for k2=1,self.neg_samples+1 do
				batch_contexts[k][k2]=0.0
			end
		end
		if self.index2word[self.train_words[i][1]] ~= nil
		then
			batch_word[count]=self.train_words[i]
			batch_contexts[count]=self.train_contexts[i]
		end
	else
		--print(self.index2word[self.train_words[i][1]])
		if self.index2word[self.train_words[i][1]] ~= nil
		then
			batch_word[count]=self.train_words[i]
			batch_contexts[count]=self.train_contexts[i]
			count = count + 1
		end
	end
	if i%100000==0 then
            print(string.format("%d words trained in %.2f seconds. Learning rate: %.4f", i, sys.clock() - start, self.lr))
	end
    end    
end

-- train the model using config parameters
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







-- Helper functions (similarity, printing, etc)


local dump = function(vec)
    vec = vec:view(vec:nElement())
    local t = {}
    for i=1,vec:nElement() do
        t[#t+1] = string.format('%.4f', vec[i])
    end
    return table.concat(t, '\t')
end

-- Row-normalize a matrix
function Word2Vec:normalize(m)
    m_norm = torch.zeros(m:size())
    for i = 1, m:size(1) do
    	m_norm[i] = m[i] / torch.norm(m[i])
    end
    return m_norm
end

function Word2Vec:print_semantic_space()
    if self.word_vecs_norm == nil then
        self.word_vecs_norm = self:normalize(self.word_vecs.weight:double())
    end
    for word,_ in pairs(self.vocab) do
	vec=self.word_vecs_norm[self.word2index[word]]
	vec:resize(vec:size(1),1)				--can only transpose vector with dim 2, so resize
	vec=vec:t()
	io.write(word,"\t",dump(vec),"\n")
    end
end

-- Return the k-nearest words to a word or a vector based on cosine similarity
-- w can be a string such as "king" or a vector for ("king" - "queen" + "man")
function Word2Vec:get_sim_words(w, k)
    if self.word_vecs_norm == nil then
        self.word_vecs_norm = self:normalize(self.word_vecs.weight:double())
    end
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

-- print similar words
function Word2Vec:print_sim_words(words, k)
    for i = 1, #words do
    	r = self:get_sim_words(words[i], k)
	if r ~= nil then
   	    print("-------"..words[i].."-------")
	    for j = 1, k do
	        print(string.format("%s, %.4f", r[j][1], r[j][2]))
	    end
	end
    end
end


-- Functions we're not really using


-- Train on sentences that are streamed from the hard drive
-- Check train_mem function to train from memory (after pre-loading data into tensor)
function Word2Vec:train_stream(corpus)
    print("Training...")
    local start = sys.clock()
    local c = 0
    f = io.open(corpus, "r")
    for line in f:lines() do
        sentence = self:split(line)
        for i, word in ipairs(sentence) do
	    word=word:lower()
	    word_idx = self.word2index[word]
	    if word_idx ~= nil then -- word exists in vocab
    	        --local reduced_window = torch.random(self.window) -- pick random window size
		self.word[1] = word_idx -- update current word
                --for j = i - reduced_window, i + reduced_window do -- loop through contexts
                for j = i - self.window, i + self.window do -- loop through contexts
	            local context = sentence[j]
		    if context ~= nil and j ~= i then -- possible context
		        context_idx = self.word2index[context]
			if context_idx ~= nil then -- valid context
  		            self:sample_contexts(context_idx) -- update pos/neg contexts
			    self:train_pair(self.word, self.contexts) -- train word context pair
			    c = c + 1
			    self.lr = math.max(self.min_lr, self.lr + self.decay) 
			    if c % 100000 ==0 then
			        print(string.format("%d words trained in %.2f seconds. Learning rate: %.4f", c, sys.clock() - start, self.lr))
			    end
			end
		    end
                end		
	    end
	end
    end
end


-- Build a table of unigram probabilities from which to obtain negative samples (from background corpus, not current!!)
-- FIX: this doesn't work: what do we do with the words which don't appear in the background corpus?

--function Word2Vec:build_table_background()
--   local start = sys.clock()
--   local word2freq = {}
--   local total_count = 0
--   print("Building a table of unigram probabilities from background corpus... ")
--   for line in io.lines(self.background_freqs) do
--   	local parts = split(line, "\t")
--	--print(parts[1],parts[2])
--   	local word = parts[2]
--	local freq = tonumber(parts[1])
--	total_count = total_count + freq^self.alpha
--	if self.word2index[word] ~= nil								-- only record words we need, otherwise memory problems!
--	then
--		word2freq[word] = freq^self.alpha
--	end
--   end
--   for idx = 1, self.vocab_size do
--        word_prob = word2freq[self.index2word[idx]]^self.alpha / total_count			-- computing word probability
--        self.index2prob[idx] = word_prob	-- assign probability to idx		
--	print(self.index2word[idx],word2freq[self.index2word[idx]],word_prob)
--   end
--   print(string.format("Done in %.2f seconds.", sys.clock() - start))
--end
