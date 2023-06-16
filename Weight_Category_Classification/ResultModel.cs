using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLnetBeginner.Weight_Category_Classification
{
    internal class ResultModel
    {
        [ColumnName("PredictedLabel")]
        public int Prediction { get; set; }

        public float[] Score { get; set; }  
    }
}
