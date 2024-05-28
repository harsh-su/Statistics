## Copyright (C) 2024 Ruchika Sonagote <ruchikasonagote2003@gmail.com>
##
## This file is part of the statistics package for GNU Octave.
##
## This program is free software; you can redistribute it and/or modify it under
## the terms of the GNU General Public License as published by the Free Software
## Foundation; either version 3 of the License, or (at your option) any later
## version.
##
## This program is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
## FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
## details.
##
## You should have received a copy of the GNU General Public License along with
## this program; if not, see <http://www.gnu.org/licenses/>.


classdef ClassificationPartitionedModel

	properties
		## BinEdges 							= [];
		## CategoricalPredictors 	= [];
		ClassNames 								= [];
		Cost 											= [];
		CrossValidatedModel 			= [];
		KFold 										= [];
		ModelParameters 					= [];
		NumObservations 					= [];
		Partition 								= [];
		PredictorNames 						= [];
		Prior 										= [];
		ResponseName 							= [];
		## ScoreTransform 				= [];
		Trained 									= [];
		## W 											= [];
		X 												= [];
		Y 												= [];
	endproperties

	methods (Access = public)

		## Constructor to initialize the partitioned model
		function this = ClassificationPartitionedModel (Mdl, Partition)
			this.X = Mdl.X;
			this.Y = Mdl.Y;
			this.KFold = get (Partition, "NumTestSets");
			this.ClassNames = Mdl.ClassNames;
			this.Prior = Mdl.Prior;
			this.Cost = Mdl.Cost;
			this.ResponseName = Mdl.ResponseName;
			this.NumObservations = Mdl.NumObservations;
			this.PredictorNames = Mdl.PredictorNames;
			this.Trained = cell (get (Partition, "NumTestSets"), 1);
			this.Partition = Partition;
			this.CrossValidatedModel = class (Mdl);
			
			switch (this.CrossValidatedModel)
				case 'ClassificationKNN'
					## Arguments to pass in fitcknn
					args = {};
					## List of acceptable parameters for fitcknn
					acceptableParams = {...
						'PredictorNames', 'ResponseName', 'BreakTies', 'NSMethod', ...
						'BucketSize', 'Cost', 'Prior', 'NumNeighbors', 'Exponent', ...
						'Scale', 'Cov', 'Distance', 'DistanceWeight', 'IncludeTies'};

					for i = 1:numel (acceptableParams)
						paramName = acceptableParams{i};
						if (isprop (Mdl, paramName))
							paramValue = Mdl.(paramName);
							if (!isempty (paramValue))
								args = [args, {paramName, paramValue}];
							endif
						else
							switch paramName
								case 'Cov'
									if (strcmpi (Mdl.Distance, 'mahalanobis') && !isempty (Mdl.DistParameter))
										args = [args, {'Cov', Mdl.DistParameter}];
									endif
								case 'Exponent'
									if (strcmpi (Mdl.Distance, 'minkowski') && !isempty (Mdl.DistParameter))
										args = [args, {'Exponent', Mdl.DistParameter}];
									endif
								case 'Scale'
									if (strcmpi (Mdl.Distance, 'seuclidean') && !isempty (Mdl.DistParameter))
										args = [args, {'Scale', Mdl.DistParameter}];
									endif
							endswitch
						endif
					endfor

					## Train models on k-1 folds and reserve 1 fold for validation
					for k = 1:this.KFold
						trainIdx = training (this.Partition, k);
						this.Trained{k} = fitcknn (this.X(trainIdx, :), this.Y(trainIdx), args{:});
					endfor
					
					## State the ModelParameters
					params = struct();
					paramList = {'NumNeighbors', 'Distance', 'DistParameter', 'NSMethod', 'DistanceWeight', 'Standardize'};
					for i = 1:numel (paramList)
						paramName = paramList{i};
						if (isprop (Mdl, paramName))
							params.(paramName) = Mdl.(paramName);
						endif
					endfor
					this.ModelParameters = params;
				otherwise
					error ("ClassificationPartitionedModel: Unsupported Model Type");												
				endswitch
		endfunction

		## Function to predict on out-of-fold data
		function [label, Score, Cost] = kfoldPredict (this)

			## Initialize the label vector based on the type of Y
			if iscellstr (this.Y)
				label = cell (this.NumObservations, 1);
			elseif islogical (this.Y)
				label = false (this.NumObservations, 1);
			elseif isnumeric (this.Y)
				label = zeros (this.NumObservations, 1);
			elseif ischar (this.Y)
				label = char (zeros (this.NumObservations, size (this.Y, 2)));
			else
				error ('ClassificationPartitionedModel.kfoldPredict: Unsupported data type for Y');
			endif

			Score = nan (this.NumObservations, numel (this.ClassNames));	
			Cost = nan (this.NumObservations, numel (this.ClassNames));

			switch (this.CrossValidatedModel)
				case 'ClassificationKNN'
					for k = 1:this.KFold
						testIdx = test (this.Partition, k);
						model = this.Trained{k};

						[predictedLabel, score, cost] = predict (model, this.X(testIdx, :));
						
						## Convert cell array of labels to appropriate type
						if (iscell (predictedLabel))
							if (isnumeric (this.Y))
								predictedLabel = cellfun (@str2num, predictedLabel);
							elseif (ischar (this.Y) || isstring (this.Y))
								predictedLabel = string (predictedLabel);
							elseif (islogical (this.Y))
								predictedLabel = cellfun (@logical, predictedLabel);
							elseif (iscellstr (this.Y))
								predictedLabel = predictedLabel;
							endif
						endif
						
						label(testIdx) = predictedLabel;
						Score(testIdx, :) = score;

						if nargout > 2
							Cost(testIdx, :) = cost;
						endif
					endfor
					
					## Handle single fold case (holdout)
					if this.KFold == 1
						trainIdx = training (this.Partition, 1);
						label(trainIdx) = mode (this.Y);
						Score(trainIdx, :) = NaN;
						Cost(trainIdx, :) = NaN;
					endif
				otherwise
					error ("ClassificationPartitionedModel.kfoldPredict: Unsupported Model");
			endswitch

		endfunction

		## Function to compute classification loss on out-of-fold data
		function loss = kfoldLoss (this)
			predictions = this.kfoldPredict ();
			loss = sum (predictions != this.Y) / this.NumObservations;
		endfunction	
		
	endmethods

endclassdef