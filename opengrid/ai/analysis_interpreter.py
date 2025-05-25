"""Analysis Interpreter for OpenAI Responses

Parses and structures AI responses for power systems analysis.
Author: Nik Jois (nikjois@llamasearch.ai)
License: MIT
"""

import re
import structlog
from typing import Dict, List, Any
from dataclasses import dataclass

logger = structlog.get_logger(__name__)

@dataclass
class ParsedAnalysis:
    """Parsed analysis response structure"""
    summary: str
    insights: List[str]
    recommendations: List[str]
    warnings: List[str]
    confidence_score: float
    metadata: Dict[str, Any]

class AnalysisInterpreter:
    """Interprets and structures AI analysis responses"""
    
    def __init__(self):
        """Initialize the analysis interpreter"""
        self.section_patterns = {
            'summary': [
                r'(?:executive\s+)?summary:?(.+?)(?=key\s+insights|insights|recommendations|\n\n|\Z)',
                r'overview:?(.+?)(?=key\s+insights|insights|recommendations|\n\n|\Z)',
                r'assessment:?(.+?)(?=key\s+insights|insights|recommendations|\n\n|\Z)'
            ],
            'insights': [
                r'(?:key\s+)?insights?:?(.+?)(?=recommendations|warnings|next\s+steps|\n\n|\Z)',
                r'key\s+findings:?(.+?)(?=recommendations|warnings|next\s+steps|\n\n|\Z)',
                r'analysis\s+results:?(.+?)(?=recommendations|warnings|next\s+steps|\n\n|\Z)'
            ],
            'recommendations': [
                r'recommendations?:?(.+?)(?=warnings|concerns|next\s+steps|\n\n|\Z)',
                r'actions?:?(.+?)(?=warnings|concerns|next\s+steps|\n\n|\Z)',
                r'solutions?:?(.+?)(?=warnings|concerns|next\s+steps|\n\n|\Z)'
            ],
            'warnings': [
                r'warnings?:?(.+?)(?=next\s+steps|conclusions?|\n\n|\Z)',
                r'concerns?:?(.+?)(?=next\s+steps|conclusions?|\n\n|\Z)',
                r'risks?:?(.+?)(?=next\s+steps|conclusions?|\n\n|\Z)',
                r'cautions?:?(.+?)(?=next\s+steps|conclusions?|\n\n|\Z)'
            ]
        }
        
        logger.info("AnalysisInterpreter initialized")
    
    def parse_response(self, content: str) -> ParsedAnalysis:
        """Parse AI response content into structured format
        
        Args:
            content: Raw AI response content
            
        Returns:
            Parsed analysis structure
        """
        try:
            # Clean the content
            cleaned_content = self._clean_content(content)
            
            # Extract sections
            summary = self._extract_section(cleaned_content, 'summary')
            insights = self._extract_list_section(cleaned_content, 'insights')
            recommendations = self._extract_list_section(cleaned_content, 'recommendations')
            warnings = self._extract_list_section(cleaned_content, 'warnings')
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(cleaned_content, summary, insights, recommendations)
            
            # Extract metadata
            metadata = self._extract_metadata(cleaned_content)
            
            return ParsedAnalysis(
                summary=summary,
                insights=insights,
                recommendations=recommendations,
                warnings=warnings,
                confidence_score=confidence_score,
                metadata=metadata
            )
            
        except Exception as e:
            logger.warning("Error parsing AI response", error=str(e))
            return self._create_fallback_response(content, str(e))
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content"""
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r'\s+', ' ', content)
        
        # Normalize section headers
        content = re.sub(r'(\d+\.?\s*)', '', content)  # Remove numbering
        content = content.lower()
        
        return content.strip()
    
    def _extract_section(self, content: str, section_type: str) -> str:
        """Extract a single section from content"""
        patterns = self.section_patterns.get(section_type, [])
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                section_text = match.group(1).strip()
                # Clean up the section
                section_text = re.sub(r'^\s*[-•*]\s*', '', section_text, flags=re.MULTILINE)
                section_text = re.sub(r'\n+', ' ', section_text)
                return section_text[:1000]  # Limit length
        
        # Fallback: try to find any occurrence of the section type
        fallback_pattern = rf'{section_type}:?\s*(.{{1,500}})'
        match = re.search(fallback_pattern, content, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()[:500]
        
        return f"No {section_type} section found"
    
    def _extract_list_section(self, content: str, section_type: str) -> List[str]:
        """Extract a list-based section from content"""
        patterns = self.section_patterns.get(section_type, [])
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                section_text = match.group(1).strip()
                return self._parse_list_items(section_text)
        
        # Fallback
        fallback_pattern = rf'{section_type}:?\s*(.{{1,1000}})'
        match = re.search(fallback_pattern, content, re.IGNORECASE | re.DOTALL)
        if match:
            return self._parse_list_items(match.group(1).strip())
        
        return []
    
    def _parse_list_items(self, text: str) -> List[str]:
        """Parse text into list items"""
        # Split by common list delimiters
        items = []
        
        # Try numbered lists first
        numbered_items = re.findall(r'\d+\.?\s*([^0-9\n]+)', text)
        if numbered_items:
            items = [item.strip() for item in numbered_items]
        else:
            # Try bulleted lists
            bulleted_items = re.findall(r'[-•*]\s*([^\n•*-]+)', text)
            if bulleted_items:
                items = [item.strip() for item in bulleted_items]
            else:
                # Try sentence-based splitting
                sentences = re.split(r'[.!]\s+', text)
                items = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # Clean and filter items
        cleaned_items = []
        for item in items[:10]:  # Limit to 10 items
            item = item.strip()
            if len(item) > 5 and len(item) < 200:  # Reasonable length
                cleaned_items.append(item)
        
        return cleaned_items
    
    def _calculate_confidence_score(self, content: str, summary: str, insights: List[str], recommendations: List[str]) -> float:
        """Calculate confidence score based on response completeness"""
        score = 0.0
        
        # Base score for having content
        if len(content) > 100:
            score += 0.3
        
        # Score for having summary
        if summary and "no" not in summary.lower() and len(summary) > 20:
            score += 0.3
        
        # Score for having insights
        if insights and len(insights) >= 2:
            score += 0.2
        
        # Score for having recommendations
        if recommendations and len(recommendations) >= 2:
            score += 0.2
        
        # Check for technical terms (indicates domain expertise)
        technical_terms = [
            'voltage', 'current', 'power', 'frequency', 'impedance', 'reactance',
            'transformer', 'generator', 'load', 'fault', 'protection', 'stability',
            'harmonic', 'thd', 'power flow', 'contingency', 'mw', 'mva', 'kv',
            'per unit', 'pu', 'ieee', 'nerc', 'reliability'
        ]
        
        technical_count = sum(1 for term in technical_terms if term in content.lower())
        technical_score = min(0.2, technical_count * 0.02)
        score += technical_score
        
        # Penalty for error indicators
        error_terms = ['error', 'failed', 'unknown', 'n/a', 'not available']
        error_count = sum(1 for term in error_terms if term in content.lower())
        score -= min(0.3, error_count * 0.1)
        
        return max(0.0, min(1.0, score))
    
    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from content"""
        metadata = {}
        
        # Extract numerical values
        numbers = re.findall(r'(\d+\.?\d*)\s*(%|mw|mv|kv|ka|pu|degrees?)', content, re.IGNORECASE)
        if numbers:
            metadata['extracted_values'] = [{'value': float(n[0]), 'unit': n[1]} for n in numbers[:5]]
        
        # Extract voltage levels
        voltages = re.findall(r'(\d+\.?\d*)\s*kv', content, re.IGNORECASE)
        if voltages:
            metadata['voltage_levels'] = [float(v) for v in voltages[:5]]
        
        # Extract percentages
        percentages = re.findall(r'(\d+\.?\d*)\s*%', content)
        if percentages:
            metadata['percentages'] = [float(p) for p in percentages[:5]]
        
        # Check for compliance standards
        standards = re.findall(r'(ieee\s*\d+|iec\s*\d+|nerc\s*\w+)', content, re.IGNORECASE)
        if standards:
            metadata['standards_mentioned'] = standards[:3]
        
        # Extract severity indicators
        if any(word in content for word in ['critical', 'emergency', 'severe']):
            metadata['severity_level'] = 'high'
        elif any(word in content for word in ['warning', 'caution', 'concern']):
            metadata['severity_level'] = 'medium'
        else:
            metadata['severity_level'] = 'low'
        
        # Response length indicator
        metadata['response_length'] = len(content)
        metadata['word_count'] = len(content.split())
        
        return metadata
    
    def _create_fallback_response(self, content: str, error: str) -> ParsedAnalysis:
        """Create fallback response when parsing fails"""
        # Extract first few sentences as summary
        sentences = re.split(r'[.!?]\s+', content)
        summary = '. '.join(sentences[:2]) if sentences else content[:200]
        
        # Try to extract any bullet points as insights
        bullets = re.findall(r'[-•*]\s*([^\n]+)', content)
        insights = bullets[:3] if bullets else []
        
        return ParsedAnalysis(
            summary=summary,
            insights=insights,
            recommendations=["Review analysis manually due to parsing issues"],
            warnings=[f"Response parsing failed: {error}"],
            confidence_score=0.3,
            metadata={
                'parse_error': error,
                'fallback_used': True,
                'content_length': len(content)
            }
        )
    
    def validate_analysis(self, analysis: ParsedAnalysis) -> Dict[str, Any]:
        """Validate parsed analysis for completeness and quality"""
        validation = {
            'is_valid': True,
            'issues': [],
            'quality_score': 0.0
        }
        
        # Check summary quality
        if not analysis.summary or len(analysis.summary) < 20:
            validation['issues'].append("Summary too short or missing")
            validation['is_valid'] = False
        else:
            validation['quality_score'] += 0.3
        
        # Check insights
        if not analysis.insights or len(analysis.insights) < 2:
            validation['issues'].append("Insufficient insights provided")
        else:
            validation['quality_score'] += 0.3
        
        # Check recommendations
        if not analysis.recommendations:
            validation['issues'].append("No recommendations provided")
        else:
            validation['quality_score'] += 0.3
        
        # Check confidence score
        if analysis.confidence_score < 0.5:
            validation['issues'].append("Low confidence in analysis")
        else:
            validation['quality_score'] += 0.1
        
        # Check for technical content
        all_text = f"{analysis.summary} {' '.join(analysis.insights)} {' '.join(analysis.recommendations)}"
        technical_terms = ['voltage', 'power', 'current', 'mw', 'kv', 'transformer', 'generator']
        has_technical = any(term in all_text.lower() for term in technical_terms)
        
        if not has_technical:
            validation['issues'].append("Lacks technical power systems content")
        else:
            validation['quality_score'] += 0.1
        
        validation['quality_score'] = min(1.0, validation['quality_score'])
        
        return validation
    
    def extract_action_items(self, analysis: ParsedAnalysis) -> List[Dict[str, Any]]:
        """Extract actionable items from analysis"""
        action_items = []
        
        # Extract from recommendations
        for i, rec in enumerate(analysis.recommendations):
            priority = 'medium'
            if any(word in rec.lower() for word in ['immediate', 'urgent', 'critical']):
                priority = 'high'
            elif any(word in rec.lower() for word in ['consider', 'future', 'long-term']):
                priority = 'low'
            
            action_items.append({
                'id': f'rec_{i+1}',
                'description': rec,
                'type': 'recommendation',
                'priority': priority,
                'source': 'ai_analysis'
            })
        
        # Extract from warnings
        for i, warning in enumerate(analysis.warnings):
            action_items.append({
                'id': f'warn_{i+1}',
                'description': warning,
                'type': 'warning',
                'priority': 'high',
                'source': 'ai_analysis'
            })
        
        return action_items
    
    def generate_executive_summary(self, analyses: List[ParsedAnalysis]) -> Dict[str, Any]:
        """Generate executive summary from multiple analyses"""
        summary = {
            'total_analyses': len(analyses),
            'average_confidence': sum(a.confidence_score for a in analyses) / len(analyses) if analyses else 0,
            'key_themes': [],
            'critical_issues': [],
            'top_recommendations': []
        }
        
        # Collect all text for analysis
        all_insights = []
        all_recommendations = []
        all_warnings = []
        
        for analysis in analyses:
            all_insights.extend(analysis.insights)
            all_recommendations.extend(analysis.recommendations)
            all_warnings.extend(analysis.warnings)
        
        # Find common themes (simplified keyword extraction)
        common_words = {}
        for text in all_insights + all_recommendations:
            words = re.findall(r'\b\w{4,}\b', text.lower())
            for word in words:
                common_words[word] = common_words.get(word, 0) + 1
        
        # Get top themes
        sorted_words = sorted(common_words.items(), key=lambda x: x[1], reverse=True)
        summary['key_themes'] = [word for word, count in sorted_words[:5] if count > 1]
        
        # Critical issues (from warnings)
        summary['critical_issues'] = all_warnings[:5]
        
        # Top recommendations (from highest confidence analyses)
        high_conf_analyses = sorted(analyses, key=lambda x: x.confidence_score, reverse=True)
        for analysis in high_conf_analyses[:3]:
            summary['top_recommendations'].extend(analysis.recommendations[:2])
        
        return summary


logger.info("OpenGrid analysis interpreter module loaded.") 