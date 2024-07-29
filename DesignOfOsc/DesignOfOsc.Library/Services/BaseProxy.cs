using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

#region -  版本注释 >>
/*-------------------------------------------
 * 公司名称：DesignOfOsc.Library
 * 文 件 名:  BaseProxy 
 * 创建者：xiezm
 * 创建时间：2023/9/26 15:50:47
 * 修改人：
 * 修改说明：
 * 描述：
 * 版本：V1.0.0
 *-----------------------------------------*/
#endregion
namespace DesignOfOsc.Library
{
    public class BaseProxy<T> : IDisposable where T : new()
    {
        private static T _i = default;
        private static object Singleton_Lock = new object();
        public static T i
        {
            get
            {
                if (_i == null) //双if +lock
                {
                    lock (Singleton_Lock)
                    {
                        if (_i == null)
                        {
                            _i = new T();
                        }
                    }
                }
                return _i;
            }
        }
        public void Dispose()
        {
            _i = default;
        }
    }
}
