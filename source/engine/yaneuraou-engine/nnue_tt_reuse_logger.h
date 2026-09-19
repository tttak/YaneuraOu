#pragma once
#if defined(ENABLE_NNUE_TT_REUSE_DIAGNOSTIC)
#include <cstdint>
#include <fstream>
#include <mutex>
#include <ostream>
#include <unordered_map>
#include <vector>
namespace YaneuraOu::Search::NnueTtReuseLog {
struct Lifetime { std::uint64_t key=0; std::uintptr_t slot=0; int depth=0,bound=0,abs_eval=0,instability=0; bool pv=false,evicted=false,shadow_protected=false; std::uint64_t hits=0,useful_cutoffs=0,shadow_future_misses=0; };
inline std::mutex mutex;
inline std::unordered_map<std::uintptr_t,Lifetime> live;
inline std::unordered_map<std::uint64_t,std::size_t> ghost_index;
inline std::vector<Lifetime> completed;
inline std::uint64_t probes=0,sampled_probes=0,sampled_hits=0;
inline bool Sample(std::uint64_t key){ return (key&1023)==0; }
inline int Priority(const Lifetime& x){ return x.depth+(x.pv?4:0)+(x.hits?4:0)+(x.instability>=300?2:0); }
inline void Reset(){ std::lock_guard<std::mutex> l(mutex); live.clear();ghost_index.clear();completed.clear();probes=sampled_probes=sampled_hits=0; }
inline void OnProbe(std::uint64_t key,const void* slot,bool hit){ ++probes;if(!Sample(key))return;++sampled_probes;std::lock_guard<std::mutex> l(mutex);if(hit){++sampled_hits;auto i=live.find(reinterpret_cast<std::uintptr_t>(slot));if(i!=live.end()&&i->second.key==key)++i->second.hits;ghost_index.erase(key);}else if(auto i=ghost_index.find(key);i!=ghost_index.end())++completed[i->second].shadow_future_misses; }
inline void OnUsefulCutoff(std::uint64_t key){ if(!Sample(key))return;std::lock_guard<std::mutex> l(mutex);for(auto& [slot,x]:live)if(x.key==key){++x.useful_cutoffs;break;} }
inline void OnWrite(std::uint64_t key,const void* slot,int depth,bool pv,int bound,int eval,int value,bool accepted){ if(!accepted)return;std::lock_guard<std::mutex> l(mutex);auto sid=reinterpret_cast<std::uintptr_t>(slot);auto old=live.find(sid);Lifetime in{key,sid,depth,bound,std::abs(eval),std::abs(value-eval),pv};if(old!=live.end()&&old->second.key!=key){old->second.evicted=true;old->second.shadow_protected=Priority(old->second)>Priority(in);completed.push_back(old->second);if(old->second.shadow_protected)ghost_index[old->second.key]=completed.size()-1;live.erase(old);}if(Sample(key)){auto [i,added]=live.emplace(sid,in);if(!added&&i->second.key==key){in.hits=i->second.hits;in.useful_cutoffs=i->second.useful_cutoffs;i->second=in;}} }
inline bool WriteCsv(const char* path){ std::lock_guard<std::mutex> l(mutex);std::ofstream o(path);if(!o)return false;o<<"key,slot,depth,pv,bound,abs_eval,instability,hits,useful_cutoffs,evicted,shadow_protected,shadow_future_misses\n";auto emit=[&](const Lifetime&x){o<<x.key<<','<<x.slot<<','<<x.depth<<','<<x.pv<<','<<x.bound<<','<<x.abs_eval<<','<<x.instability<<','<<x.hits<<','<<x.useful_cutoffs<<','<<x.evicted<<','<<x.shadow_protected<<','<<x.shadow_future_misses<<'\n';};for(auto&x:completed)emit(x);for(auto&[s,x]:live)emit(x);return true; }
inline void Report(std::ostream&o){ std::lock_guard<std::mutex> l(mutex);std::uint64_t reused=0,useful=0,protected_count=0,recovered=0;auto add=[&](const Lifetime&x){reused+=x.hits>0;useful+=x.useful_cutoffs>0;protected_count+=x.shadow_protected;recovered+=x.shadow_future_misses;};for(auto&x:completed)add(x);for(auto&[s,x]:live)add(x);o<<"NNUE TT reuse diagnostic begin\nprobes "<<probes<<"\nsampled_probes "<<sampled_probes<<"\nsampled_hits "<<sampled_hits<<"\nlifetimes "<<completed.size()+live.size()<<"\nreused_lifetimes "<<reused<<"\nuseful_lifetimes "<<useful<<"\nshadow_protected "<<protected_count<<"\nshadow_future_misses "<<recovered<<"\nNNUE TT reuse diagnostic end\n"; }
}
#endif
