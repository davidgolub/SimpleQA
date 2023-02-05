The SimpleQuestions Dataset
--------------------------------------------------------
In this directory is the SimpleQuestions dataset collected for
research in automatic question answering.

** DATA **
SimpleQuestions is a dataset for simple QA, which consists
of a total of 108,442 questions written in natural language by human
English-speaking annotators each paired with a corresponding fact,
formatted as (subject, relationship, object), that provides the answer
but also a complete explanation.  Fast have been extracted from the
Knowledge Base Freebase (freebase.com).  We randomly shuffle these
questions and use 70\% of them (75910) as training set, 10\% as
validation set (10845), and the remaining 20\% as test set.

** FORMAT **
Data is organized in 3 files: annotated_fb_data_{train, valid, test}.txt .
Each file contains one example per line with the following format:
"Subject-entity [tab] relationship [tab] Object-entity [tab] question",
with Subject-entity, relationship and Object-entity being www links
pointing to the actual Freebase entities.

** DATA COLLECTION**
We collected SimpleQuestions in two phases.  The first phase consisted
of shortlisting the set of facts from Freebase to be annotated with
questions.  We used Freebase as background KB and removed all facts
with undefined relationship type i.e. containing the word
"freebase". We also removed all facts for which the (subject,
relationship) pair had more than a threshold number of objects. This
filtering step is crucial to remove facts which would result in
trivial uninformative questions, such as, "Name a person who is an
actor?". The threshold was set to 10.

In the second phase, these selected facts were sampled and delivered
to human annotators to generate questions from them. For the sampling,
each fact was associated with a probability which defined as a
function of its relationship frequency in the KB: to favor
variability, facts with relationship appearing more
frequently were given lower probabilities.  For each sampled facts,
annotators were shown the facts along with hyperlinks to
www.freebase.com to provide some context while framing the
question. Given this information, annotators were asked to phrase a
question involving the subject and the relationship
of the fact, with the answer being the object.  The
annotators were explicitly instructed to phrase the question
differently as much as possible, if they encounter multiple facts with
similar relationship.  They were also given the option of
skipping facts if they wish to do so.  This was very important to
avoid the annotators to write a boiler plate questions when they had
no background knowledge about some facts.

