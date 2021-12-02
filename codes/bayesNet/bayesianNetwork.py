from xml.dom.minidom import parse
import xml.dom.minidom
import numpy as np
import itertools
from itertools import product
import copy
import random
 
class bayesian_network():
    def __init__(self):
        self.net = {}

    def __estr(self, e):
        evidence = ''
        i = 0
        for var, val in e.items():
            if i > 0:
                evidence += ','
            evidence += '%s=%s' % (var, val)
            i += 1
        return evidence

    def __calc_cum_prob(self, p):
        cp = [0 for _ in p]
        cp[0] = p[0]
        for i in range(1, len(p)):
            cp[i] = cp[i-1] + p[i]

        return cp

    def __topology_sort(self):
        if self.net == {}:
            return
        
        # add the variable `v iff all parents of `v` are already in.
        var = list(self.net.keys())
        self.variables = []
        while len(self.variables) < len(var):
            for v in var:
                if v not in self.variables and all(x in self.variables for x in self.net[v]['parents']):
                    self.variables.append(v)

    def __find_markov_blanket(self):
        if self.net == {}:
            return
        
        for var in self.net.keys():
            s = self.net[var]['parents'] + self.net[var]['children']
            for v in self.net[var]['children']:
                s.extend(self.net[v]['parents'])
            s = list(set(s))
            if var in s:
                s.remove(var)
            self.net[var]['markov_blanket'] = s

    def load(self, xmlbif):
        '''a common representation for Bayesian networks is called XMLBIF. More information can be found here: http://www.cs.cmu.edu/afs/cs/user/fgcozman/www/Research/InterchangeFormat/.'''
        DOMTree = xml.dom.minidom.parse(xmlbif)
        collection = DOMTree.documentElement
        for n in collection.childNodes[1].childNodes:
            if not isinstance(n, xml.dom.minidom.Element):
                continue

            if n.tagName == 'VARIABLE':
                for sn in n.childNodes:
                    if not isinstance(sn, xml.dom.minidom.Element):
                        continue

                    if sn.tagName == 'NAME':
                        name = sn.childNodes[0].data
                        self.net[name] = {'parents': [], 'children': [], 'valIdx': {}, 'values': [], 'prob': [], 'cum_prob': [], 'condprob': {}, 'cum_condprob': {}}
                    elif sn.tagName == 'OUTCOME':
                        val = sn.childNodes[0].data
                        idx = len(self.net[name]['values'])
                        self.net[name]['values'].append(val)
                        self.net[name]['valIdx'][val] = idx
            elif n.tagName == 'DEFINITION':
                for sn in n.childNodes:
                    if not isinstance(sn, xml.dom.minidom.Element):
                        continue

                    if sn.tagName == 'FOR':
                        name = sn.childNodes[0].data
                    elif sn.tagName == 'GIVEN':
                        self.net[name]['parents'].append(sn.childNodes[0].data)
                    elif sn.tagName == 'TABLE':
                        table = [float(item) for item in sn.childNodes[0].data.split(' ')]
                        if len(self.net[name]['parents']) == 0:
                            self.net[name]['prob'] = table
                        else:
                            table = np.reshape(table, [-1, len(self.net[name]['values'])])
                            pvals = product(*[self.net[v]['values'] for v in self.net[name]['parents']])
                            for i, pval in enumerate(pvals):
                                self.net[name]['condprob'][pval] = table[i,]

        for var in self.net:
            for item in self.net[var]['parents']:
                self.net[item]['children'].append(var)

            if len(self.net[var]['parents']) == 0:
                self.net[var]['cum_prob'] =  self.__calc_cum_prob(self.net[var]['prob'])
            else:
                for pval, cprob in self.net[var]['condprob'].items():
                    self.net[var]['cum_condprob'][pval] = np.array(self.__calc_cum_prob(cprob))

        self.__topology_sort()
        self.__find_markov_blanket()
        
    def normalize(self, dist):
        if sum(dist) == 0:
            return dist

        factor = 1./ sum(dist)
        return [x * factor  for x in dist]


    def querygiven(self, Y, e):
        """ P(Y | e), Y in e """
        pidx = self.net[Y]['valIdx'][e[Y]]
        if len(self.net[Y]['parents']) == 0:
            return self.net[Y]['prob'][pidx]

        parents = tuple(e[p] for p in self.net[Y]['parents'])
        return self.net[Y]['condprob'][parents][pidx]


    def enum_all(self, variables, e):
        if len(variables) == 0:
            return 1.0
        Y = variables[0]
        if Y in e:
            ret = self.querygiven(Y, e) * self.enum_all(variables[1:], e)
        else:
            probs = []
            e2 = copy.deepcopy(e)
            for y in self.net[Y]['values']:
                e2[Y] = y
                probs.append(self.querygiven(Y, e2) * self.enum_all(variables[1:], e2))
            ret = sum(probs)

        return ret

    def exact_inference(self, Y, e, quiet=False):
        """ distribution of P(Y|e)
        `Y`: query variable
        `e`: evidence dict
        if `Y` in `e`, return P(e(Y)|e'), in which `e'` is evidence that except `Y`
        """
        value = ''
        if Y in e:
            value = e.pop(Y)
            
        dist = []
        for x in self.net[Y]['values']:
            e[Y] = x
            dist.append(self.enum_all(self.variables, e))

        dist = self.normalize(dist)
        e.pop(Y)
        evidence = self.__estr(e)
        if value == '':
            if not quiet:
                string = ['P(%s=%s|%s)=%f' % (Y, self.net[Y]['values'][i], evidence, p) \
                    for i, p in enumerate(dist)]
                print('; '.join(string))
            return dist
        else:
            if not quiet:
                string = 'P(%s=%s|%s)=%f' % (Y, value, evidence, dist[self.net[Y]['valIdx'][value]])
                print(string)
            return dist[self.net[Y]['valIdx'][value]]

    def sample(self, Y, e):
        p = random.random()
        if Y not in e:
            if len(self.net[Y]['parents']) == 0:
                for i in range(len(self.net[Y]['values'])):
                    if p <= self.net[Y]['cum_prob'][i]:
                        e[Y] = self.net[Y]['values'][i]
                        break
            else:
                parents = tuple(e[p] for p in self.net[Y]['parents'])
                for i in range(len(self.net[Y]['values'])):
                    if p <= self.net[Y]['cum_condprob'][parents][i]:
                        e[Y] = self.net[Y]['values'][i]
                        break

        return e

    def reject_sample(self, Y, e, N=int(1e6)):
        value = e.pop(Y) if Y in e else ''

        cnt = [0 for _ in self.net[Y]['values']]
        dist = self.normalize(cnt)
        dist_history = []
        for i in range(N):
            example = {}
            reject = False
            for j, var in enumerate(self.variables):
                example = self.sample(var, example)
                if var in e:
                    if example[var] != e[var]:
                        reject = True
                        break

            if not reject:
                cnt[self.net[Y]['valIdx'][example[Y]]] += 1
                dist = self.normalize(cnt)

            if value != '':
                dist_history.append(dist[self.net[Y]['valIdx'][value]])
            else:
                dist_history.append(dist)

        return dist_history

    def gibbs_sample(self, Y, e, N=int(1e6)):
        value = ''
        if Y in e:
            value = e.pop(Y)

        cnt = [0 for _ in range(len(self.net[Y]['values']))]
        dist_history = []
        example = {}
        cnodes = []
        for var in self.variables:
            if var in e:
                example[var] = e[var]
            else:
                cnodes.append(var)
                example[var] = self.net[var]['values'][0]
        
        L = len(cnodes)
        for i in range(N):
            idx = i % L
            var = cnodes[idx]

            parents_val = tuple([example[p] for p in self.net[var]['parents']])
            probs = copy.deepcopy(self.net[var]['condprob'][parents_val]) if len(parents_val) > 0 else copy.deepcopy(self.net[var]['prob'])
            for mvar in self.net[var]['markov_blanket']:
                if len(self.net[mvar]['parents']) == 0:
                    factor = self.net[mvar]['prob'][self.net[mvar]['valIdx'][example[mvar]]]
                    for j in range(len(probs)):
                        probs[j] *= factor
                else:
                    for j in range(len(probs)):
                        parents_val = tuple([example[p] if p != var else self.net[p]['values'][j] for p in self.net[mvar]['parents']])
                        factor = self.net[mvar]['condprob'][parents_val][self.net[mvar]['valIdx'][example[mvar]]] if len(parents_val) > 0 else self.net[mvar]['prob'][self.net[mvar]['valIdx'][example[mvar]]]
                        probs[j] *= factor

            probs = self.normalize(probs)
            cum_probs = self.__calc_cum_prob(probs)

            p = random.random()
            for j in range(len(cum_probs)):
                if p <= cum_probs[j]:
                    example[var] = self.net[var]['values'][j]
                    break

            cnt[self.net[Y]['valIdx'][example[Y]]] += 1
            dist = self.normalize(cnt)
            if value != '':
                dist_history.append(dist[self.net[Y]['valIdx'][value]])
            else:
                dist_history.append(dist)

        return dist_history



net_alarm = bayesian_network()
net_alarm.load('alarm.xmlbif')


def test_querygiven_alarm():
    cases = [
        (('burglary', {'burglary': 'false'}), .999),
        (('burglary', {'burglary': 'true'}), .001),
        (('alarm', {'alarm': 'high', 'burglary': 'true'}), .94002),
        (('alarm', {'alarm': 'high', 'burglary': 'true', 'earthquake': 'false'}), .94),
        (('alarm', {'alarm': 'low', 'burglary': 'false', 'earthquake': 'true'}), .71),
        (('alarm', {'alarm': 'low', 'burglary': 'false', 'earthquake': 'false'}), .999),
        (('marycalls', {'marycalls': 'false', 'alarm': 'low'}), .99)
    ]
    for i, o in cases:
        assert(abs(net_alarm.exact_inference(*i, quiet=True) - o) < 1e-5)
    print('test query passed')

def test_alarm_ask():
    inputs = [
        ('burglary', {'johncalls': 'false', 'marycalls': 'true'}),
        ('alarm', {'burglary': 'true', 'earthquake': 'false', 'johncalls': 'true', 'marycalls': 'true'}),
        ('marycalls', {'burglary': 'false', 'earthquake': 'false'}),
        ('earthquake', {'burglary': 'true'}),
        ('earthquake', {'alarm': 'high', 'marycalls': 'false'})
    ]
    outputs = [
        (0.0069, 0.9931),
        (0.9999, 0.0001),
        (0.0107, 0.9893),
        (0.0020, 0.9980),
        (0.2310, 0.7690)
    ]
    for i, i1 in enumerate(inputs):
        res = net_alarm.exact_inference(*i1, quiet=True)
        assert(abs(res[0] - outputs[i][0]) < 1e-4)
        assert(abs(res[1] - outputs[i][1]) < 1e-4)

    print('test enumeration algorithm passed')

test_querygiven_alarm()
test_alarm_ask()
print()

network = bayesian_network()
network.load('network.xmlbif')

# casual query
query = ('job', {'grade': 'high'})
network.exact_inference(*query)

# diagnostic query
query = ('grade', {'job': 'bad'})
network.exact_inference(*query)

# sanity  query
query = ('recommend_letter', {'smart': 'true'})
network.exact_inference(*query)

# sanity  query
query = ('recommend_letter', {'smart': 'true', 'effort': 'false'})
network.exact_inference(*query)

# casual query
query = ('job', {'grade': 'high', 'internship': 'true'})
network.exact_inference(*query)

print()

# sampling
import matplotlib.pyplot as plt
gt = net_alarm.exact_inference('marycalls', {'marycalls': 'false', 'alarm': 'low'}, quiet=True)
dist_history1 = net_alarm.reject_sample('marycalls', {'marycalls': 'false', 'alarm': 'low'})
dist_history2 = net_alarm.gibbs_sample('marycalls', {'marycalls': 'false', 'alarm': 'low'})
plt.plot(dist_history1)
plt.plot(dist_history2)
plt.xlabel('number of samples')
plt.ylabel('p(marycalls=false|alarm=low)')
plt.legend(['reject sampling', 'gibbs sampling'])
plt.show()

rel_err1 = [abs(gt-i) / gt for i in dist_history1]
rel_err2 = [abs(gt-i) / gt for i in dist_history2]
plt.plot(rel_err1)
plt.plot(rel_err2)
plt.xlabel('number of samples')
plt.ylabel('relative error')
plt.legend(['reject sampling', 'gibbs sampling'])
plt.show()

gt = net_alarm.exact_inference('alarm', {'alarm': 'high', 'burglary': 'true', 'earthquake': 'false'}, quiet=True)
dist_history1 = net_alarm.reject_sample('alarm', {'alarm': 'high', 'burglary': 'true', 'earthquake': 'false'})
dist_history2 = net_alarm.gibbs_sample('alarm', {'alarm': 'high', 'burglary': 'true', 'earthquake': 'false'})
plt.plot(dist_history1)
plt.plot(dist_history2)
plt.xlabel('number of samples')
plt.ylabel('p(alarm=high|burglary=true,earthquake=false)')
plt.legend(['reject sampling', 'gibbs sampling'])
plt.show()

rel_err1 = [abs(gt-i) / gt for i in dist_history1]
rel_err2 = [abs(gt-i) / gt for i in dist_history2]
plt.plot(rel_err1)
plt.plot(rel_err2)
plt.xlabel('number of samples')
plt.ylabel('relative error')
plt.legend(['reject sampling', 'gibbs sampling'])
plt.show()


# sampling for my network
gt = network.exact_inference('job', {'job': 'good', 'grade': 'high', 'internship': 'true'})
dist_history1 = network.reject_sample('job', {'job': 'good', 'grade': 'high', 'internship': 'true'})
dist_history2 = network.gibbs_sample('job', {'job': 'good', 'grade': 'high', 'internship': 'true'})
plt.plot(dist_history1)
plt.plot(dist_history2)
plt.xlabel('number of samples')
plt.ylabel('p(job=good|grade=high,internship=true)')
plt.legend(['reject sampling', 'gibbs sampling'])
plt.show()

rel_err1 = [abs(gt-i) / gt for i in dist_history1]
rel_err2 = [abs(gt-i) / gt for i in dist_history2]
plt.plot(rel_err1)
plt.plot(rel_err2)
plt.xlabel('number of samples')
plt.ylabel('relative error')
plt.legend(['reject sampling', 'gibbs sampling'])
plt.show()


gt = network.exact_inference('recommend_letter', {'recommend_letter': 'true', 'smart': 'true'})
dist_history1 = network.reject_sample('recommend_letter', {'recommend_letter': 'true', 'smart': 'true'})
dist_history2 = network.gibbs_sample('recommend_letter', {'recommend_letter': 'true', 'smart': 'true'})
plt.plot(dist_history1)
plt.plot(dist_history2)
plt.xlabel('number of samples')
plt.ylabel('p(recommend_letter=true|smart=true)')
plt.legend(['reject sampling', 'gibbs sampling'])
plt.show()

rel_err1 = [abs(gt-i) / gt for i in dist_history1]
rel_err2 = [abs(gt-i) / gt for i in dist_history2]
plt.plot(rel_err1)
plt.plot(rel_err2)
plt.xlabel('number of samples')
plt.ylabel('relative error')
plt.legend(['reject sampling', 'gibbs sampling'])
plt.show()


gt = network.exact_inference('grade', {'grade': 'high', 'job': 'bad'})
dist_history1 = network.reject_sample('grade', {'grade': 'high', 'job': 'bad'})
dist_history2 = network.gibbs_sample('grade', {'grade': 'high', 'job': 'bad'})
plt.plot(dist_history1)
plt.plot(dist_history2)
plt.xlabel('number of samples')
plt.ylabel('p(grade=high|job=bad)')
plt.legend(['reject sampling', 'gibbs sampling'])
plt.show()

rel_err1 = [abs(gt-i) / gt for i in dist_history1]
rel_err2 = [abs(gt-i) / gt for i in dist_history2]
plt.plot(rel_err1)
plt.plot(rel_err2)
plt.xlabel('number of samples')
plt.ylabel('relative error')
plt.legend(['reject sampling', 'gibbs sampling'])
plt.show()
