using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLnetBeginner.Weight_Category_Classification
{
    internal class InputModel
    {
        [LoadColumn(0)]
        public string Gender { get; set; }

        [LoadColumn(1)]
        public float Height { get; set; }

        [LoadColumn(2)]
        public float Weight { get; set; }

        [LoadColumn(3)]
        public int Index { get; set; }
    }
}
