--[[
Class for word2vec with skipgram and negative sampling
--]]

require("sys")
require("nn")
require 'util.backgroundSpace'

local Word2Vec = torch.class("Word2Vec")

function Word2Vec:__init(config)
    self.tensortype = torch.getdefaulttensortype()
    self.gpu = config.gpu -- 1 if train on gpu, otherwise cpu
    self.stream = config.stream -- 1 if stream from hard drive, 0 otherwise
    self.neg_samples = config.neg_samples
    self.minfreq = config.minfreq
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


------------------------------------------------
-- Initialise model
------------------------------------------------

function Word2Vec:initialise_model(background)
    self.original_lr = self.lr
    if background=="none" then
        -- initialize word/context embeddings now that vocab size is known
        self.word_vecs = nn.LookupTable(self.vocab_size, self.dim) -- word embeddings
        self.context_vecs = nn.LookupTable(self.vocab_size, self.dim) -- context embeddings
    else
       self.word_vecs = backgroundSpace(self.word2index, self.dim, background) -- word embeddings from background
       self.context_vecs = backgroundSpace(self.word2index, self.dim, background) -- word embeddings from background
    end
    --In original code, the context vectors are set to 0
    self.context_vecs.weight=self.context_vecs.weight*0.0
    self.word_vecs.weight:apply(function() return (torch.random()/(2^32)-0.5)/self.dim end) 
    
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
end 


-----------------------------------------------------------------------
-- Build vocab frequency, word2index, and index2word from input file
-----------------------------------------------------------------------

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
    --print(self.index2word)
end



--------------------------------------------------------------------------------
-- Build a table of unigram probabilities from which to obtain negative samples
--------------------------------------------------------------------------------

function Word2Vec:build_table_current()
    local start = sys.clock()
    local total_count_pow = 0
    print("Building a table of unigram probabilities... ")
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
    print(string.format("Done in %.2f seconds.", sys.clock() - start))
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
-- Check train_mem function to train from memory (after pre-loading data into tensor)
----------------------------------------------------------------------------------------

function Word2Vec:train_stream(corpus)
    print("Training (streaming)...")
    local start = sys.clock()
    local c = 0				-- word count
    local total_c = 0			-- total word count

    local BUFSIZE = 2^15    
    local f = io.input(corpus)   -- open input file
    while true do
	--print("Buffering...")
	local lines, rest = f:read(BUFSIZE, "*line")
	if not lines then break end
	if rest then lines = lines .. rest .. '\n' end
	--print("Finished buffering...")

	start = sys.clock()
	sentence = self:split(lines)
	total_c=total_c+#sentence
	local subsampled_line=""
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
	--print("Subsampled line: ",subsampled_line)

	local batch_word = torch.zeros(self.batch_size,1)
	local batch_contexts = torch.zeros(self.batch_size,self.neg_samples+1)

	sentence = self:split(subsampled_line)
	len_sentence=#sentence
	local reduced_window = torch.random(self.window) -- pick random window size
	--print(#sentence,reduced_window)
	count = 1
	for i,word in ipairs(sentence) do
		--print(i,word)
		word_idx = self.word2index[word]
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
					self:train_pair(batch_word, batch_contexts)
					-- Original from w2v c: alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
					-- Original from torch implementation: 
					self.lr = math.max(self.min_lr, self.lr + self.decay)
					--print(self.original_lr,c,self.epochs,self.total_count)
					--new_lr = self.original_lr*(1-c/(self.epochs * self.total_count +1))
					--if new_lr > self.min_lr then
					--	self.lr=new_lr
					--end
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
				-- Also cater for end of sentence
				--if i == len_sentence and j == len_sentence - 1 then
				--	print(count)
				--	print(batch_word,batch_contexts)
				--	self:train_pair(batch_word, batch_contexts)
				--	self.lr = math.max(self.min_lr, self.lr + self.decay)
				--end
						
				c = c + 1
				--if c % 10000 ==0 then
				
				    --self:print_semantic_space()
				--end
			end
		    end
		end		
	end
	print(string.format("%d word pairs trained in %.2f seconds. Learning rate: %.4f", c, sys.clock() - start, self.lr))
	print("Total count:",total_c)
	self.loss_toprint=self.loss_toprint/c
	print(string.format("Loss: %.7f",self.loss_toprint))
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

--------------------------------------------------
-- Print semantic space in DISSECT .dm format
--------------------------------------------------

function Word2Vec:print_semantic_space()
    self.word_vecs_norm = self:normalize(self.word_vecs.weight:double())
    for word,_ in pairs(self.vocab) do
	vec=self.word_vecs_norm[self.word2index[word]]
	vec:resize(vec:size(1),1)				--can only transpose vector with dim 2, so resize
	vec=vec:t()
	io.write(word,"\t",dump(vec),"\n")
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



----------------------------------------------------------------
-- Backup function: training from memory.
-- At the moment, I am not using this because putting the data
-- onto the gpu is much too slow
---------------------------------------------------------------

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
	    print(string.format("Loss: %.4f",self.loss_toprint/100000))
	    self.loss_toprint=0.0
	    self:print_semantic_space()
	end
    end    
end



-- pre-load data as a torch tensor instead of streaming it. this requires a lot of memory, 
-- so if the corpus is huge you should partition into smaller sets
function Word2Vec:preload_data(corpus)
    print("Preloading training corpus into tensors (Warning: this takes a lot of memory)")
    local start = sys.clock()
    local c = 0
    f = io.open(corpus, "r")
    self.train_words = {}; self.train_contexts = {}
    for line in f:lines() do										-- loop through each line in corpus
        sentence = self:split(line)									-- split line into words
	local subsampled_line= ""
        for i, word in ipairs(sentence) do								-- for each word...
	    word=word:lower()										-- convert to lower case...
	    word_idx = self.word2index[word]								-- get the index of the word...
	    freq=self.vocab[self.index2word[word_idx]]
	    if freq ~= nil then -- word exists in vocab
		--subsampl=(torch.sqrt(freq/(10^-4*self.total_count))+1)*(10^-4*self.total_count)/freq
		p_keep=(torch.sqrt(self.subsampl/(freq/self.total_count)))	--original eq from Mikolov 2013 paper
		-- (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn -- eq from word2vec c code
		--p_keep = (torch.sqrt(freq / (self.subsampl * self.total_count)) + 1) * (self.subsampl * self.total_count) / freq
		random=torch.random()/(2^32)
		--print(freq,self.total_count)
		--print(p_keep,random)
		if p_keep > random then
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
				    --self:sample_contexts(context_idx) -- update pos/neg contexts		-- get negative contexts (this has the effect of filling self.contexts with neg context ids)
  		                    self:sample_contexts(context_idx,word_idx) -- update pos/neg contexts
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
