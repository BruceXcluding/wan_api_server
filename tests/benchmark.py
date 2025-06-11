#!/usr/bin/env python3
"""
简洁的性能基准测试
测试I2V服务的基本性能指标
"""

import sys
import time
import json
import asyncio
import argparse
from pathlib import Path

# 设置路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import aiohttp
import requests

class SimpleBenchmark:
    def __init__(self, url="http://localhost:8088"):
        self.url = url.rstrip('/')
        self.results = []
        
    def get_test_data(self):
        """获取测试数据"""
        return {
            "prompt": "A cat walking in the garden",
            "image_url": "https://picsum.photos/512/512",
            "image_size": "512*512",
            "num_frames": 25,  # 少帧数快速测试
            "seed": 42
        }
    
    def check_health(self):
        """健康检查"""
        try:
            response = requests.get(f"{self.url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Service: {data.get('device_type')} x {data.get('device_count')}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Service not accessible: {e}")
            return False
    
    async def single_request(self, session):
        """单个请求测试"""
        start_time = time.time()
        
        try:
            # 提交任务
            async with session.post(
                f"{self.url}/submit",
                json=self.get_test_data(),
                timeout=30
            ) as response:
                if response.status != 200:
                    return False, time.time() - start_time, f"Submit failed: {response.status}"
                
                result = await response.json()
                task_id = result.get("task_id")
                
                if not task_id:
                    return False, time.time() - start_time, "No task ID"
            
            # 等待完成
            max_wait = 300  # 5分钟超时
            while time.time() - start_time < max_wait:
                async with session.get(f"{self.url}/status/{task_id}", timeout=10) as resp:
                    if resp.status == 200:
                        status_data = await resp.json()
                        status = status_data.get("status", "unknown")
                        
                        if status == "completed":
                            return True, time.time() - start_time, "Success"
                        elif status in ["failed", "error"]:
                            return False, time.time() - start_time, f"Task failed: {status}"
                        
                        await asyncio.sleep(2)
                    else:
                        await asyncio.sleep(1)
            
            return False, time.time() - start_time, "Timeout"
            
        except Exception as e:
            return False, time.time() - start_time, f"Error: {e}"
    
    async def response_time_test(self, count=3):
        """响应时间测试"""
        print(f"\n🚀 Response Time Test ({count} requests)")
        print("-" * 40)
        
        times = []
        success_count = 0
        
        async with aiohttp.ClientSession() as session:
            for i in range(count):
                print(f"Request {i+1}/{count}...", end=" ")
                
                success, duration, message = await self.single_request(session)
                times.append(duration)
                
                if success:
                    success_count += 1
                    print(f"✅ {duration:.1f}s")
                else:
                    print(f"❌ {duration:.1f}s ({message})")
        
        # 计算统计
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            success_rate = (success_count / count) * 100
            
            print(f"\n📊 Results:")
            print(f"   Success Rate: {success_rate:.1f}% ({success_count}/{count})")
            print(f"   Avg Time:     {avg_time:.1f}s")
            print(f"   Min Time:     {min_time:.1f}s")
            print(f"   Max Time:     {max_time:.1f}s")
            
            return {
                "test": "response_time",
                "requests": count,
                "success_count": success_count,
                "success_rate": success_rate,
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time
            }
        
        return {"test": "response_time", "error": "No valid responses"}
    
    async def concurrent_test(self, users=2, requests_per_user=2):
        """并发测试"""
        print(f"\n🔥 Concurrent Test ({users} users, {requests_per_user} requests each)")
        print("-" * 50)
        
        total_requests = users * requests_per_user
        all_times = []
        total_success = 0
        
        start_time = time.time()
        
        async def user_task(user_id):
            user_times = []
            user_success = 0
            
            async with aiohttp.ClientSession() as session:
                for req in range(requests_per_user):
                    success, duration, message = await self.single_request(session)
                    user_times.append(duration)
                    
                    if success:
                        user_success += 1
                        print(f"User {user_id+1}-{req+1}: ✅ {duration:.1f}s")
                    else:
                        print(f"User {user_id+1}-{req+1}: ❌ {duration:.1f}s")
            
            return user_success, user_times
        
        # 并发执行
        tasks = [user_task(i) for i in range(users)]
        results = await asyncio.gather(*tasks)
        
        # 汇总结果
        for user_success, user_times in results:
            total_success += user_success
            all_times.extend(user_times)
        
        total_time = time.time() - start_time
        
        if all_times:
            success_rate = (total_success / total_requests) * 100
            avg_time = sum(all_times) / len(all_times)
            throughput = total_success / total_time
            
            print(f"\n📊 Concurrent Results:")
            print(f"   Total Requests: {total_requests}")
            print(f"   Success Rate:   {success_rate:.1f}% ({total_success}/{total_requests})")
            print(f"   Avg Time:       {avg_time:.1f}s")
            print(f"   Throughput:     {throughput:.2f} req/s")
            
            return {
                "test": "concurrent",
                "users": users,
                "total_requests": total_requests,
                "success_count": total_success,
                "success_rate": success_rate,
                "avg_time": avg_time,
                "throughput": throughput,
                "total_time": total_time
            }
        
        return {"test": "concurrent", "error": "No valid responses"}
    
    def memory_test(self):
        """内存状态检查"""
        print(f"\n💾 Memory Check")
        print("-" * 20)
        
        try:
            response = requests.get(f"{self.url}/metrics", timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # 提取关键信息
                queue_size = data.get("queue_size", 0)
                active_tasks = data.get("active_tasks", 0)
                
                print(f"✅ Queue Size:    {queue_size}")
                print(f"✅ Active Tasks:  {active_tasks}")
                
                return {
                    "test": "memory",
                    "queue_size": queue_size,
                    "active_tasks": active_tasks
                }
            else:
                print(f"❌ Metrics not available")
        except Exception as e:
            print(f"❌ Memory check failed: {e}")
        
        return {"test": "memory", "error": "Failed to get metrics"}
    
    def save_results(self, filepath="benchmark_results.json"):
        """保存结果"""
        if self.results:
            try:
                with open(filepath, 'w') as f:
                    json.dump({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "url": self.url,
                        "results": self.results
                    }, f, indent=2)
                print(f"\n💾 Results saved to {filepath}")
            except Exception as e:
                print(f"❌ Save failed: {e}")

async def main():
    parser = argparse.ArgumentParser(description="Simple I2V Benchmark")
    parser.add_argument("--url", default="http://localhost:8088", help="Service URL")
    parser.add_argument("--quick", action="store_true", help="Quick test (1 request)")
    parser.add_argument("--concurrent", action="store_true", help="Run concurrent test")
    parser.add_argument("--count", type=int, default=3, help="Number of requests")
    parser.add_argument("--users", type=int, default=2, help="Concurrent users")
    parser.add_argument("--save", help="Save results to file")
    
    args = parser.parse_args()
    
    print("🎯 FastAPI I2V Simple Benchmark")
    print("=" * 40)
    print(f"Target: {args.url}")
    
    benchmark = SimpleBenchmark(args.url)
    
    try:
        # 健康检查
        if not benchmark.check_health():
            print("💥 Service not available")
            return 1
        
        # 内存检查
        memory_result = benchmark.memory_test()
        benchmark.results.append(memory_result)
        
        # 响应时间测试
        count = 1 if args.quick else args.count
        response_result = await benchmark.response_time_test(count)
        benchmark.results.append(response_result)
        
        # 并发测试
        if args.concurrent and not args.quick:
            concurrent_result = await benchmark.concurrent_test(args.users, 2)
            benchmark.results.append(concurrent_result)
        
        # 保存结果
        if args.save:
            benchmark.save_results(args.save)
        
        print("\n🎉 Benchmark completed!")
        
        # 简单总结
        success_count = sum(r.get("success_count", 0) for r in benchmark.results if "success_count" in r)
        if success_count > 0:
            print(f"✅ Total successful requests: {success_count}")
        else:
            print("⚠️ No successful requests")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏸️ Interrupted")
        return 1
    except Exception as e:
        print(f"\n💥 Error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))